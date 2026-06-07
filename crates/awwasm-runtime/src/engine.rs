// ============================================================================
// Instruction Stream Abstraction
// ============================================================================

pub trait InstrSource<'a> {
    // Takes &self (via Cell interior mutability) so multiple shared borrows can coexist:
    // `instr` from one next_instr call and `&source` passed to dispatch_tail are both
    // shared borrows — the borrow checker allows them simultaneously.
    fn next_instr(&self) -> Option<&AwwasmInstruction<'a>>;
}

pub struct SliceInstrSource<'a, 'c> {
    slice: &'c [AwwasmInstruction<'a>],
    idx: core::cell::Cell<usize>,
}

impl<'a, 'c> SliceInstrSource<'a, 'c> {
    pub fn new(slice: &'c [AwwasmInstruction<'a>]) -> Self {
        Self { slice, idx: core::cell::Cell::new(0) }
    }
}

impl<'a, 'c> InstrSource<'a> for SliceInstrSource<'a, 'c> {
    #[inline]
    fn next_instr(&self) -> Option<&AwwasmInstruction<'a>> {
        let i = self.idx.get();
        if let Some(instr) = self.slice.get(i) {
            self.idx.set(i + 1);
            Some(instr)
        } else {
            None
        }
    }
}

// ============================================================================
// Tail-Call Dispatch Macros
// ============================================================================

#[cfg(feature = "tail_calls")]
macro_rules! dispatch_next {
    ($thread:expr, $source:expr, $frame_idx:expr) => {
        if let Some(next_instr) = $source.next_instr() {
            become $thread.dispatch_tail(next_instr, $source, $frame_idx)
        } else {
            Ok(ControlSignal::None)
        }
    }
}

#[cfg(not(feature = "tail_calls"))]
macro_rules! dispatch_next {
    ($thread:expr, $source:expr, $frame_idx:expr) => {{
        // Touch args so the compiler doesn't warn about them being unused
        // on stable (where the trampoline loop drives iteration instead).
        let _ = ($source, $frame_idx);
        Ok(ControlSignal::None)
    }}
}

#[cfg(feature = "tail_calls")]
macro_rules! handle_op {
    ($op_func:ident, $thread:expr, $source:expr, $frame_idx:expr $(, $arg:expr)*) => {{
        become $thread.$op_func($source, $frame_idx $(, $arg)*)
    }}
}

#[cfg(not(feature = "tail_calls"))]
macro_rules! handle_op {
    ($op_func:ident, $thread:expr, $source:expr, $frame_idx:expr $(, $arg:expr)*) => {{
        $thread.$op_func($source, $frame_idx $(, $arg)*)
    }}
}

// Execution engine — stack-based WebAssembly interpreter.
//
// `AwwasmThread` is the entry point: create one, call `invoke()`,
// get results back. Internally it uses the parser's `InstructionIterator`
// to stream instructions from resolved function bodies.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use alloc::rc::Rc;

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
    /// Start index of this frame's locals in `AwwasmThread::locals_pool`.
    locals_start: usize,
    /// Number of local variable slots (params + declared locals) for this frame.
    locals_count: usize,
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
    /// Flat pool for all active frames' local variables — avoids per-call heap allocation.
    /// Each `CallFrame` stores a (locals_start, locals_count) range into this Vec.
    /// On frame entry, locals are appended; on frame exit, the Vec is truncated.
    locals_pool: Vec<AwwasmValue>,
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
            locals_pool: Vec::with_capacity(256),
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

        // Collect results and release the top frame's locals
        let frame = self.call_stack.pop().unwrap();
        self.locals_pool.truncate(frame.locals_start);
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

            // Ensure function body is fully parsed (no-op after first call)
            self.ensure_fully_parsed(func_addr)?;

            // Clone the Rc to get an independent reference to the instructions,
            // then drop the store borrow so &mut self methods can be called during dispatch.
            let instrs_rc = {
                let func = self.store.func(func_addr)?;
                match func {
                    AwwasmFuncInst::Wasm(wasm) => match &wasm.code {
                        LazyResolvedCodeRef::FullyParsed { instrs, .. } => Rc::clone(instrs),
                        LazyResolvedCodeRef::Unparsed { .. } => {
                            return Err(AwwasmRuntimeError::FunctionNotParsed);
                        }
                    }
                    AwwasmFuncInst::Host(_) => {
                        return Err(AwwasmRuntimeError::HostFunctionNotExecutable);
                    }
                }
            };
            let signal = self.execute_instructions_vec(&instrs_rc, frame_idx)?;

            match signal {
                ControlSignal::Return | ControlSignal::None => {
                    if self.call_stack.len() == 1 {
                        return Ok(());
                    }
                    let frame = self.call_stack.pop().unwrap();
                    self.locals_pool.truncate(frame.locals_start);
                    self.stack.truncate(frame.stack_height + frame.arity as usize);
                }
                ControlSignal::Branch(_) => {
                    return Err(AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable));
                }
            }
        }
    }

    // ========================================================================
    // Instruction dispatch (from cached instruction Vec)
    // ========================================================================

    fn execute_instructions_vec(
        &mut self,
        instrs: &[AwwasmInstruction<'a>],
        frame_idx: usize,
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let source = SliceInstrSource::new(instrs);
        while let Some(instr) = source.next_instr() {
            let signal = self.dispatch_tail(instr, &source, frame_idx)?;
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

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn dispatch_tail<S: InstrSource<'a>>(
        &mut self,
        instr: &AwwasmInstruction<'a>,
        source: &S,
        frame_idx: usize,
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        match &instr.operands {
            AwwasmOperands::I32Const(op) => handle_op!(op_i32_const, self, source, frame_idx, op),
            AwwasmOperands::I64Const(op) => handle_op!(op_i64_const, self, source, frame_idx, op),
            AwwasmOperands::F32Const(op) => handle_op!(op_f32_const, self, source, frame_idx, op),
            AwwasmOperands::F64Const(op) => handle_op!(op_f64_const, self, source, frame_idx, op),
            AwwasmOperands::I32Add => handle_op!(op_i32_add, self, source, frame_idx),
            AwwasmOperands::I32Sub => handle_op!(op_i32_sub, self, source, frame_idx),
            AwwasmOperands::I32Mul => handle_op!(op_i32_mul, self, source, frame_idx),
            AwwasmOperands::I32Eqz => handle_op!(op_i32_eqz, self, source, frame_idx),
            AwwasmOperands::I32Eq => handle_op!(op_i32_eq, self, source, frame_idx),
            AwwasmOperands::I32Ne => handle_op!(op_i32_ne, self, source, frame_idx),
            AwwasmOperands::LocalGet(op) => handle_op!(op_local_get, self, source, frame_idx, op),
            AwwasmOperands::LocalSet(op) => handle_op!(op_local_set, self, source, frame_idx, op),
            AwwasmOperands::LocalTee(op) => handle_op!(op_local_tee, self, source, frame_idx, op),
            AwwasmOperands::GlobalGet(op) => handle_op!(op_global_get, self, source, frame_idx, op),
            AwwasmOperands::GlobalSet(op) => handle_op!(op_global_set, self, source, frame_idx, op),
            AwwasmOperands::I32Load(op) => handle_op!(op_i32_load, self, source, frame_idx, op),
            AwwasmOperands::I64Load(op) => handle_op!(op_i64_load, self, source, frame_idx, op),
            AwwasmOperands::I32Store(op) => handle_op!(op_i32_store, self, source, frame_idx, op),
            AwwasmOperands::I64Store(op) => handle_op!(op_i64_store, self, source, frame_idx, op),
            AwwasmOperands::MemorySize(op) => handle_op!(op_memory_size, self, source, frame_idx, op),
            AwwasmOperands::MemoryGrow(op) => handle_op!(op_memory_grow, self, source, frame_idx, op),
            AwwasmOperands::Call(op) => handle_op!(op_call, self, source, frame_idx, op),
            AwwasmOperands::Block(op) => handle_op!(op_block, self, source, frame_idx, op),
            AwwasmOperands::Loop(op) => handle_op!(op_loop, self, source, frame_idx, op),
            AwwasmOperands::If(op) => handle_op!(op_if, self, source, frame_idx, op),
            AwwasmOperands::Br(op) => handle_op!(op_br, self, source, frame_idx, op),
            AwwasmOperands::BrIf(op) => handle_op!(op_br_if, self, source, frame_idx, op),
            AwwasmOperands::BrTable(op) => handle_op!(op_br_table, self, source, frame_idx, op),
            AwwasmOperands::Return => handle_op!(op_return, self, source, frame_idx),
            AwwasmOperands::CallIndirect(op) => handle_op!(op_call_indirect, self, source, frame_idx, op),
            AwwasmOperands::End => handle_op!(op_end, self, source, frame_idx),
            AwwasmOperands::Else => handle_op!(op_else, self, source, frame_idx),
            AwwasmOperands::Unreachable => handle_op!(op_unreachable, self, source, frame_idx),
            AwwasmOperands::Nop => handle_op!(op_nop, self, source, frame_idx),
            AwwasmOperands::Drop => handle_op!(op_drop, self, source, frame_idx),
            AwwasmOperands::Select => handle_op!(op_select, self, source, frame_idx),

        }
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_const<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::I32ConstOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        self.stack.push(AwwasmValue::I32(op.value));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_const<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::I64ConstOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        self.stack.push(AwwasmValue::I64(op.value));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_const<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::F32ConstOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        self.stack.push(AwwasmValue::F32(op.value));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_const<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::F64ConstOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        self.stack.push(AwwasmValue::F64(op.value));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_add<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?;
        let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a.wrapping_add(b)));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_sub<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?;
        let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a.wrapping_sub(b)));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_mul<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?;
        let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a.wrapping_mul(b)));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_eqz<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(if v == 0 { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_eq<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?;
        let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(if a == b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_ne<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?;
        let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(if a != b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_local_get<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::IndexOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let locals_start = self.call_stack[frame_idx].locals_start;
        let val = self.locals_pool
            .get(locals_start + op.index as usize)
            .copied()
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
        self.stack.push(val);
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_local_set<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::IndexOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop()?;
        let locals_start = self.call_stack[frame_idx].locals_start;
        let local = self.locals_pool
            .get_mut(locals_start + op.index as usize)
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
        *local = val;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_local_tee<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::IndexOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = *self.stack.last()
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::StackOverflow))?;
        let locals_start = self.call_stack[frame_idx].locals_start;
        let local = self.locals_pool
            .get_mut(locals_start + op.index as usize)
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
        *local = val;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_global_get<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::IndexOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let module_addr = self.call_stack[frame_idx].module_addr;
        let module_inst = self.store.module(module_addr)
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
        let global_addr = module_inst.global(op.index)
            .ok_or_else(|| AwwasmRuntimeError::InvalidGlobalAddr(op.index))?;
        let global = self.store.global(global_addr)?;
        self.stack.push(global.get());
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_global_set<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::IndexOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop()?;
        let module_addr = self.call_stack[frame_idx].module_addr;
        let module_inst = self.store.module(module_addr)
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))?;
        let global_addr = module_inst.global(op.index)
            .ok_or_else(|| AwwasmRuntimeError::InvalidGlobalAddr(op.index))?;
        let global = self.store.global_mut(global_addr)?;
        global.set(val).map_err(|_| AwwasmRuntimeError::ImmutableGlobal(op.index))?;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_load<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = base.wrapping_add(op.offset);
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let mem = self.store.mem(mem_addr)?;
        let val = mem.read_i32(addr)?;
        self.stack.push(AwwasmValue::I32(val));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_load<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = base.wrapping_add(op.offset);
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let mem = self.store.mem(mem_addr)?;
        let val = mem.read_i64(addr)?;
        self.stack.push(AwwasmValue::I64(val));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_store<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_i32()?;
        let base = self.pop_i32()? as u32;
        let addr = base.wrapping_add(op.offset);
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let mem = self.store.mem_mut(mem_addr)?;
        mem.write_i32(addr, val)?;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_store<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_i64()?;
        let base = self.pop_i32()? as u32;
        let addr = base.wrapping_add(op.offset);
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let mem = self.store.mem_mut(mem_addr)?;
        mem.write_i64(addr, val)?;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_memory_size<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, _op: &awwasm_parser::components::instructions::MemoryZeroOperands<'a>
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let mem = self.store.mem(mem_addr)?;
        self.stack.push(AwwasmValue::I32(mem.size_pages() as i32));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_memory_grow<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, _op: &awwasm_parser::components::instructions::MemoryZeroOperands<'a>
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let delta = self.pop_i32()? as u32;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let mem = self.store.mem_mut(mem_addr)?;
        let result = mem.grow(delta).map(|old| old as i32).unwrap_or(-1);
        self.stack.push(AwwasmValue::I32(result));
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_call<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::CallOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let target_func_addr = self.resolve_funcidx(frame_idx, op.funcidx)?;
        self.enter_function(target_func_addr)?;
        // Run the callee to completion in the main loop
        self.run_callee()?;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_block<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::BlockOperands<'a>
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let arity = block_arity(&op.block_type);
        let saved_height = self.stack.len();
        let signal = self.execute_instructions_vec(&op.body.0, frame_idx)?;
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
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_loop<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::LoopOperands<'a>
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        loop {
            let saved_height = self.stack.len();
            let signal = self.execute_instructions_vec(&op.body.0, frame_idx)?;
            match signal {
                ControlSignal::Branch(0) => {
                    // Branch to this loop's start (continue)
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
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_if<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::IfOperands<'a>
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let cond = self.pop_i32()?;
        let arity = block_arity(&op.block_type);
        let saved_height = self.stack.len();

        let signal = if cond != 0 {
            self.execute_instructions_vec(&op.then_body.0, frame_idx)?
        } else if let Some(ref else_body) = op.else_body {
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
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_br<S: InstrSource<'a>>(
        &mut self,
        _source: &S,
        _frame_idx: usize, op: &awwasm_parser::components::instructions::BrOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        Ok(ControlSignal::Branch(op.labelidx))
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_br_if<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize, op: &awwasm_parser::components::instructions::BrOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let cond = self.pop_i32()?;
        if cond != 0 {
            return Ok(ControlSignal::Branch(op.labelidx));
        }
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_br_table<S: InstrSource<'a>>(
        &mut self,
        _source: &S,
        _frame_idx: usize, op: &awwasm_parser::components::instructions::BrTableOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let idx = self.pop_i32()? as u32;
        let target = if (idx as usize) < op.targets.len() {
            op.targets[idx as usize]
        } else {
            op.default
        };
        Ok(ControlSignal::Branch(target))
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_return<S: InstrSource<'a>>(
        &mut self,
        _source: &S,
        _frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        Ok(ControlSignal::Return)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_call_indirect<S: InstrSource<'a>>(
        &mut self,
        _source: &S,
        _frame_idx: usize, _op: &awwasm_parser::components::instructions::CallIndirectOperands
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        Err(AwwasmRuntimeError::InstructionParseError(
            "call_indirect not yet implemented".into(),
        ))
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_end<S: InstrSource<'a>>(
        &mut self,
        _source: &S,
        _frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        Ok(ControlSignal::Return)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_else<S: InstrSource<'a>>(
        &mut self,
        _source: &S,
        _frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        // Shouldn't be reached: the parser pre-parses else into IfOperands.
        Err(AwwasmRuntimeError::InstructionParseError(
            "unexpected else instruction outside of if block".into(),
        ))
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_unreachable<S: InstrSource<'a>>(
        &mut self,
        _source: &S,
        _frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        Err(AwwasmRuntimeError::Trap(AwwasmTrap::Unreachable))
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_nop<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        // Do nothing
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_drop<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        self.pop()?;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_select<S: InstrSource<'a>>(
        &mut self,
        source: &S,
        frame_idx: usize
    ) -> Result<ControlSignal, AwwasmRuntimeError> {
        let cond = self.pop_i32()?;
                let val2 = self.pop()?;
                let val1 = self.pop()?;
                self.stack.push(if cond != 0 { val1 } else { val2 });
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // Function entry
    // ========================================================================

    /// Push a new call frame for the given function.
    fn enter_function(&mut self, func_addr: AwwasmFuncAddr) -> Result<(), AwwasmRuntimeError> {
        if self.call_stack.len() >= self.max_call_depth {
            return Err(AwwasmRuntimeError::Trap(AwwasmTrap::CallStackExhausted));
        }

        // Ensure fully parsed
        self.ensure_fully_parsed(func_addr)?;

        let func = self.store.func(func_addr)?;
        match func {
            AwwasmFuncInst::Wasm(wasm) => {
                let module_addr = wasm.module;
                let type_idx = wasm.type_idx;
                let module_inst = self.store.module(module_addr)
                    .ok_or(AwwasmRuntimeError::InvalidFuncAddr(func_addr.0))?;
                let func_type = module_inst.types.get(type_idx as usize)
                    .ok_or(AwwasmRuntimeError::InvalidFuncAddr(func_addr.0))?;
                let param_count = func_type.params.len();
                let result_count = func_type.results.len();

                // Get local declarations from fully-parsed code
                let local_types = match &wasm.code {
                    LazyResolvedCodeRef::FullyParsed { locals, .. } => {
                        locals.clone()
                    }
                    _ => return Err(AwwasmRuntimeError::FunctionNotParsed),
                };

                // Pop params from value stack and move them into the flat locals pool.
                let stack_len = self.stack.len();
                if stack_len < param_count {
                    return Err(AwwasmRuntimeError::Trap(AwwasmTrap::StackOverflow));
                }
                let params_start = stack_len - param_count;
                let locals_start = self.locals_pool.len();
                // Move params directly into the pool (no intermediate Vec)
                self.locals_pool.extend(self.stack.drain(params_start..));

                // Append zero-initialized declared locals
                for decl in &local_types {
                    let vt = match decl.type_ {
                        AwwasmValueType::I32 => AwwasmValue::I32(0),
                        AwwasmValueType::I64 => AwwasmValue::I64(0),
                        AwwasmValueType::F32 => AwwasmValue::F32(0.0),
                        AwwasmValueType::F64 => AwwasmValue::F64(0.0),
                    };
                    for _ in 0..decl.count {
                        self.locals_pool.push(vt);
                    }
                }
                let locals_count = self.locals_pool.len() - locals_start;

                let stack_height = self.stack.len();

                self.call_stack.push(CallFrame {
                    func_addr,
                    module_addr,
                    locals_start,
                    locals_count,
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

        let instrs_rc = {
            let func = self.store.func(func_addr)?;
            match func {
                AwwasmFuncInst::Wasm(wasm) => match &wasm.code {
                    LazyResolvedCodeRef::FullyParsed { instrs, .. } => Rc::clone(instrs),
                    _ => return Err(AwwasmRuntimeError::FunctionNotParsed),
                }
                _ => return Err(AwwasmRuntimeError::HostFunctionNotExecutable),
            }
        };
        let _signal = self.execute_instructions_vec(&instrs_rc, frame_idx)?;

        // Pop callee frame and release its locals from the pool
        let frame = self.call_stack.pop().unwrap();
        self.locals_pool.truncate(frame.locals_start);
        self.stack.truncate(frame.stack_height + frame.arity as usize);
        Ok(())
    }

    // ========================================================================
    // Lazy resolution
    // ========================================================================

    /// Ensure a function body is fully parsed (locals + instruction Vec cached).
    /// Subsequent executions use the cached Vec directly; no re-parsing.
    fn ensure_fully_parsed(&mut self, func_addr: AwwasmFuncAddr) -> Result<(), AwwasmRuntimeError> {
        let func = self.store.func_mut(func_addr)?;
        if let AwwasmFuncInst::Wasm(wasm) = func {
            if let LazyResolvedCodeRef::Unparsed { bytes } = wasm.code {
                let (locals, code) = parse_func_body(bytes)?;
                let instrs: Result<Vec<_>, _> = InstructionIterator::new(code).collect();
                let instrs = instrs.map_err(|e| AwwasmRuntimeError::InstructionParseError(format!("{}", e)))?;
                wasm.code = LazyResolvedCodeRef::FullyParsed { locals, instrs: Rc::new(instrs) };
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
