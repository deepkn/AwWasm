//! Execution engine — stack-based WebAssembly interpreter.
//!
//! `AwwasmThread` is the entry point: create one, call `invoke()`,
//! get results back. Internally it uses the parser's `InstructionIterator`
//! to stream instructions from resolved function bodies.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::error::{AwwasmRuntimeError, AwwasmTrap};
use crate::func::{AwwasmFuncInst, LazyResolvedCodeRef};
use crate::values::{AwwasmFuncAddr, AwwasmMemAddr, AwwasmModuleAddr, AwwasmValue, AwwasmValueType};
use crate::store::AwwasmStore;

use awwasm_parser::components::instructions::{
    AwwasmInstruction, AwwasmOperands, InstructionIterator, BlockValueType,
};

// ============================================================================
// Configuration
// ============================================================================

/// Default maximum call stack depth.
const DEFAULT_MAX_CALL_DEPTH: usize = 1024;

// ============================================================================
// Control flow signal
// ============================================================================

/// Internal signal for structured control flow within a block.
enum ControlSignal {
    /// Normal execution — keep going.
    None,
    /// `br N` / `br_if N` (true) — break out of N enclosing blocks.
    Branch(u32),
    /// `return` — immediately return from the current function.
    Return,
}

// ============================================================================
// Call frame
// ============================================================================

/// One per active function invocation.
struct CallFrame {
    /// The function address in the Store.
    func_addr: AwwasmFuncAddr,
    /// The owning module instance (for resolving locals funcidx → store addr).
    module_addr: AwwasmModuleAddr,
    /// Local variables (params filled from stack, then zero-initialized locals).
    locals: Vec<AwwasmValue>,
    /// Value-stack height at frame entry (for result truncation on return).
    stack_height: usize,
    /// Number of results this function produces.
    arity: u32,
}

// ============================================================================
// Thread (execution context)
// ============================================================================

/// The execution thread — one per `invoke()` call.
///
/// Borrows the `Store` mutably for the duration of execution.
pub struct AwwasmThread<'a, 'b> {
    /// Value stack (operand stack).
    stack: Vec<AwwasmValue>,
    /// Call stack (activation frames).
    call_stack: Vec<CallFrame>,
    /// The Store.
    store: &'b mut AwwasmStore<'a>,
    /// Max call depth.
    max_call_depth: usize,
}

impl<'a, 'b> AwwasmThread<'a, 'b> {
    /// Create a new thread bound to a Store.
    pub fn new(store: &'b mut AwwasmStore<'a>) -> Self {
        Self {
            stack: Vec::with_capacity(256),
            call_stack: Vec::with_capacity(64),
            store,
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
        }
    }

    /// Invoke a function by address with the given arguments.
    ///
    /// Returns the result values (may be empty for void functions).
    pub fn invoke(
        &mut self,
        func_addr: AwwasmFuncAddr,
        args: &[AwwasmValue],
    ) -> Result<Vec<AwwasmValue>, AwwasmRuntimeError> {
        // Push args onto value stack
        for arg in args {
            self.stack.push(*arg);
        }

        // Push the initial frame
        self.enter_function(func_addr)?;

        // Run the main loop
        self.run_loop()?;

        // Collect results
        let frame = self.call_stack.pop().unwrap();
        let results: Vec<AwwasmValue> = self.stack.drain(frame.stack_height..).collect();
        Ok(results)
    }

    // ========================================================================
    // Main execution loop
    // ========================================================================

    /// The main interpreter loop. Runs until the initial call frame returns.
    fn run_loop(&mut self) -> Result<(), AwwasmRuntimeError> {
        loop {
            if self.call_stack.is_empty() {
                return Ok(());
            }

            let frame_idx = self.call_stack.len() - 1;
            let func_addr = self.call_stack[frame_idx].func_addr;

            // Ensure function body is resolved
            self.ensure_resolved(func_addr)?;

            // Get the code bytes
            let func = self.store.func(func_addr)?;
            match func {
                AwwasmFuncInst::Wasm(wasm) => {
                    match &wasm.code {
                        LazyResolvedCodeRef::Resolved { code, .. } => {
                            // Execute instructions via iterator
                            let mut iter = InstructionIterator::new(code);
                            let signal = self.execute_instructions_iter(&mut iter, frame_idx)?;

                            match signal {
                                ControlSignal::Return | ControlSignal::None => {
                                    // Function returned — pop frame
                                    if self.call_stack.len() == 1 {
                                        // Last frame — we're done
                                        return Ok(());
                                    }
                                    let frame = self.call_stack.pop().unwrap();
                                    self.stack.truncate(frame.stack_height + frame.arity as usize);
                                }
                                ControlSignal::Branch(_) => {
                                    // Should not happen at function level
                                    return Err(AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable));
                                }
                            }
                        }
                        LazyResolvedCodeRef::Unparsed { .. } => {
                            return Err(AwwasmRuntimeError::FunctionNotParsed);
                        }
                    }
                }
                AwwasmFuncInst::Host(_) => {
                    return Err(AwwasmRuntimeError::HostFunctionNotExecutable);
                }
            }
        }
    }

    // ========================================================================
    // Instruction dispatch (from iterator — top-level function body)
    // ========================================================================

    fn execute_instructions_iter(
        &mut self,
        iter: &mut InstructionIterator<'a>,
        frame_idx: usize,
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        while let Some(result) = iter.next() {
            let instr = result.map_err(|e| {
                AwwasmRuntimeError::InstructionParseError(format!("{}", e))
            })?;

            let signal = self.dispatch(&instr, frame_idx)?;
            match signal {
                ControlSignal::None => {}
                other => return Ok(other),
            }
        }
        Ok(ControlSignal::None)
    }

    // ========================================================================
    // Instruction dispatch (from pre-parsed vec — block/loop/if bodies)
    // ========================================================================

    fn execute_instructions_vec(
        &mut self,
        instrs: &[AwwasmInstruction<'a>],
        frame_idx: usize,
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        for instr in instrs {
            let signal = self.dispatch(instr, frame_idx)?;
            match signal {
                ControlSignal::None => {}
                other => return Ok(other),
            }
        }
        Ok(ControlSignal::None)
    }

    // ========================================================================
    // Single instruction dispatch
    // ========================================================================

    fn dispatch(
        &mut self,
        instr: &AwwasmInstruction<'a>,
        frame_idx: usize,
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        match &instr.operands {
            // ==============================================================
            // Constants
            // ==============================================================
            AwwasmOperands::I32Const(op) => {
                self.stack.push(AwwasmValue::I32(op.value));
            }
            AwwasmOperands::I64Const(op) => {
                self.stack.push(AwwasmValue::I64(op.value));
            }
            AwwasmOperands::F32Const(op) => {
                self.stack.push(AwwasmValue::F32(op.value));
            }
            AwwasmOperands::F64Const(op) => {
                self.stack.push(AwwasmValue::F64(op.value));
            }

            // ==============================================================
            // Arithmetic (i32)
            // ==============================================================
            AwwasmOperands::I32Add => {
                let b = self.pop_i32()?;
                let a = self.pop_i32()?;
                self.stack.push(AwwasmValue::I32(a.wrapping_add(b)));
            }
            AwwasmOperands::I32Sub => {
                let b = self.pop_i32()?;
                let a = self.pop_i32()?;
                self.stack.push(AwwasmValue::I32(a.wrapping_sub(b)));
            }
            AwwasmOperands::I32Mul => {
                let b = self.pop_i32()?;
                let a = self.pop_i32()?;
                self.stack.push(AwwasmValue::I32(a.wrapping_mul(b)));
            }

            // ==============================================================
            // Comparison (i32)
            // ==============================================================
            AwwasmOperands::I32Eqz => {
                let v = self.pop_i32()?;
                self.stack.push(AwwasmValue::I32(if v == 0 { 1 } else { 0 }));
            }
            AwwasmOperands::I32Eq => {
                let b = self.pop_i32()?;
                let a = self.pop_i32()?;
                self.stack.push(AwwasmValue::I32(if a == b { 1 } else { 0 }));
            }
            AwwasmOperands::I32Ne => {
                let b = self.pop_i32()?;
                let a = self.pop_i32()?;
                self.stack.push(AwwasmValue::I32(if a != b { 1 } else { 0 }));
            }

            // ==============================================================
            // Local variables
            // ==============================================================
            AwwasmOperands::LocalGet(op) => {
                let val = self.call_stack[frame_idx].locals
                    .get(op.index as usize)
                    .copied()
                    .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
                self.stack.push(val);
            }
            AwwasmOperands::LocalSet(op) => {
                let val = self.pop()?;
                let local = self.call_stack[frame_idx].locals
                    .get_mut(op.index as usize)
                    .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
                *local = val;
            }
            AwwasmOperands::LocalTee(op) => {
                let val = *self.stack.last()
                    .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::StackOverflow))?;
                let local = self.call_stack[frame_idx].locals
                    .get_mut(op.index as usize)
                    .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
                *local = val;
            }

            // ==============================================================
            // Global variables
            // ==============================================================
            AwwasmOperands::GlobalGet(op) => {
                let module_addr = self.call_stack[frame_idx].module_addr;
                let module_inst = self.store.module(module_addr)
                    .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
                let global_addr = module_inst.global(op.index)
                    .ok_or_else(|| AwwasmRuntimeError::InvalidGlobalAddr(op.index))?;
                let global = self.store.global(global_addr)?;
                self.stack.push(global.get());
            }
            AwwasmOperands::GlobalSet(op) => {
                let val = self.pop()?;
                let module_addr = self.call_stack[frame_idx].module_addr;
                let module_inst = self.store.module(module_addr)
                    .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
                let global_addr = module_inst.global(op.index)
                    .ok_or_else(|| AwwasmRuntimeError::InvalidGlobalAddr(op.index))?;
                let global = self.store.global_mut(global_addr)?;
                global.set(val).map_err(|_| AwwasmRuntimeError::ImmutableGlobal(op.index))?;
            }

            // ==============================================================
            // Memory operations
            // ==============================================================
            AwwasmOperands::I32Load(memarg) => {
                let base = self.pop_i32()? as u32;
                let addr = base.wrapping_add(memarg.offset);
                let mem_addr = self.resolve_mem(frame_idx, 0)?;
                let mem = self.store.mem(mem_addr)?;
                let val = mem.read_i32(addr)?;
                self.stack.push(AwwasmValue::I32(val));
            }
            AwwasmOperands::I64Load(memarg) => {
                let base = self.pop_i32()? as u32;
                let addr = base.wrapping_add(memarg.offset);
                let mem_addr = self.resolve_mem(frame_idx, 0)?;
                let mem = self.store.mem(mem_addr)?;
                let val = mem.read_i64(addr)?;
                self.stack.push(AwwasmValue::I64(val));
            }
            AwwasmOperands::I32Store(memarg) => {
                let val = self.pop_i32()?;
                let base = self.pop_i32()? as u32;
                let addr = base.wrapping_add(memarg.offset);
                let mem_addr = self.resolve_mem(frame_idx, 0)?;
                let mem = self.store.mem_mut(mem_addr)?;
                mem.write_i32(addr, val)?;
            }
            AwwasmOperands::I64Store(memarg) => {
                let val = self.pop_i64()?;
                let base = self.pop_i32()? as u32;
                let addr = base.wrapping_add(memarg.offset);
                let mem_addr = self.resolve_mem(frame_idx, 0)?;
                let mem = self.store.mem_mut(mem_addr)?;
                mem.write_i64(addr, val)?;
            }
            AwwasmOperands::MemorySize(_) => {
                let mem_addr = self.resolve_mem(frame_idx, 0)?;
                let mem = self.store.mem(mem_addr)?;
                self.stack.push(AwwasmValue::I32(mem.size_pages() as i32));
            }
            AwwasmOperands::MemoryGrow(_) => {
                let delta = self.pop_i32()? as u32;
                let mem_addr = self.resolve_mem(frame_idx, 0)?;
                let mem = self.store.mem_mut(mem_addr)?;
                let result = mem.grow(delta).map(|old| old as i32).unwrap_or(-1);
                self.stack.push(AwwasmValue::I32(result));
            }

            // ==============================================================
            // Control flow — call
            // ==============================================================
            AwwasmOperands::Call(op) => {
                let target_func_addr = self.resolve_funcidx(frame_idx, op.funcidx)?;
                self.enter_function(target_func_addr)?;
                // Run the callee to completion in the main loop
                self.run_callee()?;
            }

            // ==============================================================
            // Control flow — block
            // ==============================================================
            AwwasmOperands::Block(block_op) => {
                let arity = block_arity(&block_op.block_type);
                let saved_height = self.stack.len();
                let signal = self.execute_instructions_vec(&block_op.body.0, frame_idx)?;
                match signal {
                    ControlSignal::Branch(0) => {
                        // Branch to this block's end — truncate stack to saved + arity
                        self.truncate_stack(saved_height, arity);
                    }
                    ControlSignal::Branch(n) => {
                        return Ok(ControlSignal::Branch(n - 1));
                    }
                    ControlSignal::Return => return Ok(ControlSignal::Return),
                    ControlSignal::None => {}
                }
            }

            // ==============================================================
            // Control flow — loop
            // ==============================================================
            AwwasmOperands::Loop(loop_op) => {
                // In a loop, `br 0` jumps back to the loop start
                loop {
                    let saved_height = self.stack.len();
                    let signal = self.execute_instructions_vec(&loop_op.body.0, frame_idx)?;
                    match signal {
                        ControlSignal::Branch(0) => {
                            // Branch back to loop start — discard results, restart
                            self.stack.truncate(saved_height);
                            continue;
                        }
                        ControlSignal::Branch(n) => {
                            return Ok(ControlSignal::Branch(n - 1));
                        }
                        ControlSignal::Return => return Ok(ControlSignal::Return),
                        ControlSignal::None => break,
                    }
                }
            }

            // ==============================================================
            // Control flow — if/else
            // ==============================================================
            AwwasmOperands::If(if_op) => {
                let cond = self.pop_i32()?;
                let arity = block_arity(&if_op.block_type);
                let saved_height = self.stack.len();

                let signal = if cond != 0 {
                    self.execute_instructions_vec(&if_op.then_body.0, frame_idx)?
                } else if let Some(ref else_body) = if_op.else_body {
                    self.execute_instructions_vec(&else_body.0, frame_idx)?
                } else {
                    ControlSignal::None
                };

                match signal {
                    ControlSignal::Branch(0) => {
                        self.truncate_stack(saved_height, arity);
                    }
                    ControlSignal::Branch(n) => {
                        return Ok(ControlSignal::Branch(n - 1));
                    }
                    ControlSignal::Return => return Ok(ControlSignal::Return),
                    ControlSignal::None => {}
                }
            }

            // ==============================================================
            // Control flow — branches
            // ==============================================================
            AwwasmOperands::Br(op) => {
                return Ok(ControlSignal::Branch(op.labelidx));
            }
            AwwasmOperands::BrIf(op) => {
                let cond = self.pop_i32()?;
                if cond != 0 {
                    return Ok(ControlSignal::Branch(op.labelidx));
                }
            }
            AwwasmOperands::BrTable(op) => {
                let idx = self.pop_i32()? as u32;
                let target = if (idx as usize) < op.targets.len() {
                    op.targets[idx as usize]
                } else {
                    op.default
                };
                return Ok(ControlSignal::Branch(target));
            }

            // ==============================================================
            // Control flow — return / end
            // ==============================================================
            AwwasmOperands::Return => {
                return Ok(ControlSignal::Return);
            }

            // End opcode — should only appear at function body end
            // (block/loop/if handle their own `end` via many_till)
            // At function level, end means return.

            // call_indirect — not yet implemented
            AwwasmOperands::CallIndirect(_) => {
                return Err(AwwasmRuntimeError::InstructionParseError(
                    "call_indirect not yet implemented".into(),
                ));
            }

            // End — at function level, signals return
            AwwasmOperands::End => {
                return Ok(ControlSignal::Return);
            }

            // Else — should not appear standalone; it's consumed by If parsing
            AwwasmOperands::Else => {
                // If we hit this, the parser gave us an unexpected Else.
                // In the pre-parsed Block/If model this shouldn't happen.
                return Err(AwwasmRuntimeError::InstructionParseError(
                    "unexpected else instruction outside of if block".into(),
                ));
            }

            // Parametric instructions
            AwwasmOperands::Unreachable => {
                return Err(AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable));
            }
            AwwasmOperands::Nop => {
                // Do nothing
            }
            AwwasmOperands::Drop => {
                self.pop()?;
            }
            AwwasmOperands::Select => {
                let cond = self.pop_i32()?;
                let val2 = self.pop()?;
                let val1 = self.pop()?;
                self.stack.push(if cond != 0 { val1 } else { val2 });
            }
        }

        Ok(ControlSignal::None)
    }

    // ========================================================================
    // Function entry
    // ========================================================================

    /// Push a new call frame for the given function.
    fn enter_function(&mut self, func_addr: AwwasmFuncAddr) -> Result<(), AwwasmRuntimeError> {
        if self.call_stack.len() >= self.max_call_depth {
            return Err(AwwasmRuntimeError::Trap(AwwasmTrap::CallStackExhausted));
        }

        // Ensure resolved
        self.ensure_resolved(func_addr)?;

        let func = self.store.func(func_addr)?;
        match func {
            AwwasmFuncInst::Wasm(wasm) => {
                let module_addr = wasm.module;
                let param_count = wasm.func_type.params.len();
                let result_count = wasm.func_type.results.len();

                // Get local declarations from resolved code
                let local_types = match &wasm.code {
                    LazyResolvedCodeRef::Resolved { locals, .. } => {
                        locals.clone()
                    }
                    _ => return Err(AwwasmRuntimeError::FunctionNotParsed),
                };

                // Pop params from value stack (they were pushed by the caller)
                // Params are on the stack in order: first param pushed first (bottom).
                let stack_len = self.stack.len();
                if stack_len < param_count {
                    return Err(AwwasmRuntimeError::Trap(AwwasmTrap::StackOverflow));
                }
                let params_start = stack_len - param_count;
                let param_values: Vec<AwwasmValue> = self.stack.drain(params_start..).collect();

                // Build locals: first the params, then zero-initialized declared locals
                let mut locals = param_values;
                for decl in &local_types {
                    let vt = match decl.type_ {
                        AwwasmValueType::I32 => AwwasmValue::I32(0),
                        AwwasmValueType::I64 => AwwasmValue::I64(0),
                        AwwasmValueType::F32 => AwwasmValue::F32(0.0),
                        AwwasmValueType::F64 => AwwasmValue::F64(0.0),
                    };
                    for _ in 0..decl.count {
                        locals.push(vt);
                    }
                }

                let stack_height = self.stack.len();

                self.call_stack.push(CallFrame {
                    func_addr,
                    module_addr,
                    locals,
                    stack_height,
                    arity: result_count as u32,
                });

                Ok(())
            }
            AwwasmFuncInst::Host(_) => {
                Err(AwwasmRuntimeError::HostFunctionNotExecutable)
            }
        }
    }

    /// Run a callee function that was just entered via `enter_function`.
    /// This executes the callee to completion and pops its frame.
    fn run_callee(&mut self) -> Result<(), AwwasmRuntimeError> {
        let callee_depth = self.call_stack.len();
        let frame_idx = callee_depth - 1;
        let func_addr = self.call_stack[frame_idx].func_addr;

        let func = self.store.func(func_addr)?;
        match func {
            AwwasmFuncInst::Wasm(wasm) => {
                match &wasm.code {
                    LazyResolvedCodeRef::Resolved { code, .. } => {
                        let mut iter = InstructionIterator::new(code);
                        let _signal = self.execute_instructions_iter(&mut iter, frame_idx)?;

                        // Pop callee frame
                        let frame = self.call_stack.pop().unwrap();
                        self.stack.truncate(frame.stack_height + frame.arity as usize);
                        Ok(())
                    }
                    _ => Err(AwwasmRuntimeError::FunctionNotParsed),
                }
            }
            _ => Err(AwwasmRuntimeError::HostFunctionNotExecutable),
        }
    }

    // ========================================================================
    // Lazy resolution
    // ========================================================================

    /// Ensure a function body is resolved (parsed from bytes into locals + code).
    fn ensure_resolved(&mut self, func_addr: AwwasmFuncAddr) -> Result<(), AwwasmRuntimeError> {
        let func = self.store.func_mut(func_addr)?;
        if let AwwasmFuncInst::Wasm(wasm) = func {
            if let LazyResolvedCodeRef::Unparsed { bytes } = &wasm.code {
                let (locals, code) = parse_func_body(bytes)?;
                wasm.code = LazyResolvedCodeRef::Resolved { locals, code };
            }
        }
        Ok(())
    }

    // ========================================================================
    // Stack helpers
    // ========================================================================

    #[inline]
    fn pop(&mut self) -> Result<AwwasmValue, AwwasmRuntimeError> {
        self.stack.pop().ok_or(AwwasmRuntimeError::Trap(AwwasmTrap::StackOverflow))
    }

    #[inline]
    fn pop_i32(&mut self) -> Result<i32, AwwasmRuntimeError> {
        match self.pop()? {
            AwwasmValue::I32(v) => Ok(v),
            other => Err(AwwasmRuntimeError::TypeMismatch {
                expected: "i32".into(),
                got: format!("{:?}", other.value_type()),
            }),
        }
    }

    #[inline]
    fn pop_i64(&mut self) -> Result<i64, AwwasmRuntimeError> {
        match self.pop()? {
            AwwasmValue::I64(v) => Ok(v),
            other => Err(AwwasmRuntimeError::TypeMismatch {
                expected: "i64".into(),
                got: format!("{:?}", other.value_type()),
            }),
        }
    }

    /// Truncate stack to `base + arity` (keeping top `arity` values).
    fn truncate_stack(&mut self, base: usize, arity: u32) {
        if arity == 0 {
            self.stack.truncate(base);
        } else {
            let arity = arity as usize;
            let current_len = self.stack.len();
            if current_len > base + arity {
                // Move the top `arity` values down to `base`
                let start = current_len - arity;
                for i in 0..arity {
                    self.stack[base + i] = self.stack[start + i];
                }
                self.stack.truncate(base + arity);
            }
        }
    }

    // ========================================================================
    // Index resolution helpers
    // ========================================================================

    /// Resolve a funcidx (module-local) to a Store func addr.
    fn resolve_funcidx(
        &self,
        frame_idx: usize,
        funcidx: u32,
    ) -> Result<AwwasmFuncAddr, AwwasmRuntimeError> {
        let module_addr = self.call_stack[frame_idx].module_addr;
        let module_inst = self.store.module(module_addr)
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
        module_inst.func(funcidx)
            .ok_or(AwwasmRuntimeError::InvalidFuncAddr(funcidx))
    }

    /// Resolve memidx to a Store mem addr.
    fn resolve_mem(
        &self,
        frame_idx: usize,
        memidx: u32,
    ) -> Result<AwwasmMemAddr, AwwasmRuntimeError> {
        let module_addr = self.call_stack[frame_idx].module_addr;
        let module_inst = self.store.module(module_addr)
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
        module_inst.mem(memidx)
            .ok_or(AwwasmRuntimeError::InvalidMemAddr(memidx))
    }
}

// ============================================================================
// Function body parsing (resolves raw bytes → locals + code)
// ============================================================================

/// Parse the raw `func_body` bytes into local declarations and the
/// remaining code bytes.
///
/// Layout: [local_count: leb128] [local_decl]* [code bytes...]
/// Each local_decl: [count: leb128] [type: u8]
fn parse_func_body<'a>(
    bytes: &'a [u8],
) -> Result<(Vec<crate::func::AwwasmLocalDecl>, &'a [u8]), AwwasmRuntimeError> {
    use crate::func::AwwasmLocalDecl;

    let mut pos = 0;

    // Read local declaration count
    let (local_decl_count, consumed) = read_leb128_u32(bytes)
        .map_err(|_| AwwasmRuntimeError::InstructionParseError("invalid local count".into()))?;
    pos += consumed;

    let mut locals = Vec::with_capacity(local_decl_count as usize);

    for _ in 0..local_decl_count {
        // Read count
        let (count, consumed) = read_leb128_u32(&bytes[pos..])
            .map_err(|_| AwwasmRuntimeError::InstructionParseError("invalid local decl count".into()))?;
        pos += consumed;

        // Read type byte
        if pos >= bytes.len() {
            return Err(AwwasmRuntimeError::InstructionParseError("truncated local decl".into()));
        }
        let type_byte = bytes[pos];
        pos += 1;

        let type_ = match type_byte {
            0x7F => AwwasmValueType::I32,
            0x7E => AwwasmValueType::I64,
            0x7D => AwwasmValueType::F32,
            0x7C => AwwasmValueType::F64,
            _ => {
                return Err(AwwasmRuntimeError::InstructionParseError(
                    format!("unknown local type: 0x{:02X}", type_byte),
                ));
            }
        };

        locals.push(AwwasmLocalDecl { count, type_ });
    }

    // Remaining bytes are the instruction code
    let code = &bytes[pos..];
    Ok((locals, code))
}

/// Read a LEB128-encoded u32 from the given bytes.
/// Returns (value, bytes_consumed).
fn read_leb128_u32(bytes: &[u8]) -> Result<(u32, usize), ()> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;

    for (i, &byte) in bytes.iter().enumerate() {
        let low7 = (byte & 0x7F) as u32;
        result |= low7 << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 35 {
            return Err(());
        }
    }
    Err(())
}

// ============================================================================
// Block type helper
// ============================================================================

/// Determine the arity (number of result values) for a block type.
fn block_arity(bt: &BlockValueType) -> u32 {
    match bt {
        BlockValueType::VOID => 0,
        _ => 1, // I32, I64, F32, F64 all produce one value
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imports::AwwasmImports;
    use awwasm_parser::components::module::AwwasmModule;

    /// Helper: compile WAT, parse, instantiate, invoke exported function.
    fn run_wat(wat: &str, func_name: &str, args: &[AwwasmValue]) -> Result<Vec<AwwasmValue>, AwwasmRuntimeError> {
        let wasm = wat::parse_str(wat).expect("invalid WAT");
        let mut module = AwwasmModule::new(&wasm).expect("parse failed");
        if module.sections.is_some() {
            module.resolve_all_sections().expect("resolve failed");
        }
        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();
        let module_addr = store.store_init(&module, &mut imports)
            .expect("instantiation failed");

        let module_inst = store.module(module_addr).unwrap();
        let export = module_inst.export_by_str(func_name)
            .expect("export not found");
        let func_addr = match export.addr {
            crate::values::AwwasmExternAddr::Func(addr) => addr,
            _ => panic!("export is not a function"),
        };

        let mut thread = AwwasmThread::new(&mut store);
        thread.invoke(func_addr, args)
    }

    #[test]
    fn test_minimal_return() {
        let result = run_wat(
            "(module (func (export \"f\") (result i32) (i32.const 42)))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(42)]);
    }

    #[test]
    fn test_arithmetic() {
        let result = run_wat(
            "(module (func (export \"f\") (result i32)
                (i32.add (i32.const 10) (i32.const 32))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(42)]);
    }

    #[test]
    fn test_arithmetic_sub() {
        let result = run_wat(
            "(module (func (export \"f\") (result i32)
                (i32.sub (i32.const 50) (i32.const 8))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(42)]);
    }

    #[test]
    fn test_arithmetic_mul() {
        let result = run_wat(
            "(module (func (export \"f\") (result i32)
                (i32.mul (i32.const 6) (i32.const 7))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(42)]);
    }

    #[test]
    fn test_local_variables() {
        let result = run_wat(
            "(module (func (export \"f\") (result i32)
                (local i32)
                (local.set 0 (i32.const 99))
                (local.get 0)))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(99)]);
    }

    #[test]
    fn test_local_tee() {
        let result = run_wat(
            "(module (func (export \"f\") (result i32)
                (local i32)
                (local.tee 0 (i32.const 55))
                ;; local.tee leaves value on stack AND sets local
                ))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(55)]);
    }

    #[test]
    fn test_i32_eqz() {
        let result = run_wat(
            "(module (func (export \"f\") (result i32) (i32.eqz (i32.const 0))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(1)]);

        let result2 = run_wat(
            "(module (func (export \"f\") (result i32) (i32.eqz (i32.const 5))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result2, vec![AwwasmValue::I32(0)]);
    }

    #[test]
    fn test_block_branch() {
        // block { i32.const 42; br 0; i32.const 99 } → 42
        let result = run_wat(
            "(module (func (export \"f\") (result i32)
                (block (result i32)
                    (i32.const 42)
                    (br 0)
                    (i32.const 99))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(42)]);
    }

    #[test]
    fn test_if_then_else() {
        let result_true = run_wat(
            "(module (func (export \"f\") (result i32)
                (if (result i32) (i32.const 1)
                    (then (i32.const 10))
                    (else (i32.const 20)))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result_true, vec![AwwasmValue::I32(10)]);

        let result_false = run_wat(
            "(module (func (export \"f\") (result i32)
                (if (result i32) (i32.const 0)
                    (then (i32.const 10))
                    (else (i32.const 20)))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result_false, vec![AwwasmValue::I32(20)]);
    }

    #[test]
    fn test_simple_loop() {
        // Count from 0 to 5 using a loop
        let result = run_wat(
            "(module (func (export \"f\") (result i32)
                (local $i i32)
                (local.set $i (i32.const 0))
                (block $break
                    (loop $continue
                        (local.set $i (i32.add (local.get $i) (i32.const 1)))
                        (br_if $break (i32.eq (local.get $i) (i32.const 5)))
                        (br $continue)))
                (local.get $i)))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(5)]);
    }

    #[test]
    fn test_function_call() {
        let result = run_wat(
            "(module
                (func $add (param i32 i32) (result i32)
                    (i32.add (local.get 0) (local.get 1)))
                (func (export \"f\") (result i32)
                    (call $add (i32.const 20) (i32.const 22))))",
            "f",
            &[],
        ).unwrap();
        assert_eq!(result, vec![AwwasmValue::I32(42)]);
    }
}
