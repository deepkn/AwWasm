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
            AwwasmOperands::F32Load(op) => handle_op!(op_f32_load, self, source, frame_idx, op),
            AwwasmOperands::F64Load(op) => handle_op!(op_f64_load, self, source, frame_idx, op),
            AwwasmOperands::I32Load8S(op) => handle_op!(op_i32_load8_s, self, source, frame_idx, op),
            AwwasmOperands::I32Load8U(op) => handle_op!(op_i32_load8_u, self, source, frame_idx, op),
            AwwasmOperands::I32Load16S(op) => handle_op!(op_i32_load16_s, self, source, frame_idx, op),
            AwwasmOperands::I32Load16U(op) => handle_op!(op_i32_load16_u, self, source, frame_idx, op),
            AwwasmOperands::I64Load8S(op) => handle_op!(op_i64_load8_s, self, source, frame_idx, op),
            AwwasmOperands::I64Load8U(op) => handle_op!(op_i64_load8_u, self, source, frame_idx, op),
            AwwasmOperands::I64Load16S(op) => handle_op!(op_i64_load16_s, self, source, frame_idx, op),
            AwwasmOperands::I64Load16U(op) => handle_op!(op_i64_load16_u, self, source, frame_idx, op),
            AwwasmOperands::I64Load32S(op) => handle_op!(op_i64_load32_s, self, source, frame_idx, op),
            AwwasmOperands::I64Load32U(op) => handle_op!(op_i64_load32_u, self, source, frame_idx, op),
            AwwasmOperands::I32Store(op) => handle_op!(op_i32_store, self, source, frame_idx, op),
            AwwasmOperands::I64Store(op) => handle_op!(op_i64_store, self, source, frame_idx, op),
            AwwasmOperands::F32Store(op) => handle_op!(op_f32_store, self, source, frame_idx, op),
            AwwasmOperands::F64Store(op) => handle_op!(op_f64_store, self, source, frame_idx, op),
            AwwasmOperands::I32Store8(op) => handle_op!(op_i32_store8, self, source, frame_idx, op),
            AwwasmOperands::I32Store16(op) => handle_op!(op_i32_store16, self, source, frame_idx, op),
            AwwasmOperands::I64Store8(op) => handle_op!(op_i64_store8, self, source, frame_idx, op),
            AwwasmOperands::I64Store16(op) => handle_op!(op_i64_store16, self, source, frame_idx, op),
            AwwasmOperands::I64Store32(op) => handle_op!(op_i64_store32, self, source, frame_idx, op),
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

            // i32 comparisons
            AwwasmOperands::I32LtS => handle_op!(op_i32_lt_s, self, source, frame_idx),
            AwwasmOperands::I32LtU => handle_op!(op_i32_lt_u, self, source, frame_idx),
            AwwasmOperands::I32GtS => handle_op!(op_i32_gt_s, self, source, frame_idx),
            AwwasmOperands::I32GtU => handle_op!(op_i32_gt_u, self, source, frame_idx),
            AwwasmOperands::I32LeS => handle_op!(op_i32_le_s, self, source, frame_idx),
            AwwasmOperands::I32LeU => handle_op!(op_i32_le_u, self, source, frame_idx),
            AwwasmOperands::I32GeS => handle_op!(op_i32_ge_s, self, source, frame_idx),
            AwwasmOperands::I32GeU => handle_op!(op_i32_ge_u, self, source, frame_idx),
            // i32 unary
            AwwasmOperands::I32Clz    => handle_op!(op_i32_clz, self, source, frame_idx),
            AwwasmOperands::I32Ctz    => handle_op!(op_i32_ctz, self, source, frame_idx),
            AwwasmOperands::I32Popcnt => handle_op!(op_i32_popcnt, self, source, frame_idx),
            // i32 div/rem
            AwwasmOperands::I32DivS => handle_op!(op_i32_div_s, self, source, frame_idx),
            AwwasmOperands::I32DivU => handle_op!(op_i32_div_u, self, source, frame_idx),
            AwwasmOperands::I32RemS => handle_op!(op_i32_rem_s, self, source, frame_idx),
            AwwasmOperands::I32RemU => handle_op!(op_i32_rem_u, self, source, frame_idx),
            // i32 bitwise/shift/rotate
            AwwasmOperands::I32And  => handle_op!(op_i32_and, self, source, frame_idx),
            AwwasmOperands::I32Or   => handle_op!(op_i32_or, self, source, frame_idx),
            AwwasmOperands::I32Xor  => handle_op!(op_i32_xor, self, source, frame_idx),
            AwwasmOperands::I32Shl  => handle_op!(op_i32_shl, self, source, frame_idx),
            AwwasmOperands::I32ShrS => handle_op!(op_i32_shr_s, self, source, frame_idx),
            AwwasmOperands::I32ShrU => handle_op!(op_i32_shr_u, self, source, frame_idx),
            AwwasmOperands::I32Rotl => handle_op!(op_i32_rotl, self, source, frame_idx),
            AwwasmOperands::I32Rotr => handle_op!(op_i32_rotr, self, source, frame_idx),

            // i64 comparisons
            AwwasmOperands::I64Eqz  => handle_op!(op_i64_eqz, self, source, frame_idx),
            AwwasmOperands::I64Eq   => handle_op!(op_i64_eq, self, source, frame_idx),
            AwwasmOperands::I64Ne   => handle_op!(op_i64_ne, self, source, frame_idx),
            AwwasmOperands::I64LtS  => handle_op!(op_i64_lt_s, self, source, frame_idx),
            AwwasmOperands::I64LtU  => handle_op!(op_i64_lt_u, self, source, frame_idx),
            AwwasmOperands::I64GtS  => handle_op!(op_i64_gt_s, self, source, frame_idx),
            AwwasmOperands::I64GtU  => handle_op!(op_i64_gt_u, self, source, frame_idx),
            AwwasmOperands::I64LeS  => handle_op!(op_i64_le_s, self, source, frame_idx),
            AwwasmOperands::I64LeU  => handle_op!(op_i64_le_u, self, source, frame_idx),
            AwwasmOperands::I64GeS  => handle_op!(op_i64_ge_s, self, source, frame_idx),
            AwwasmOperands::I64GeU  => handle_op!(op_i64_ge_u, self, source, frame_idx),
            // i64 unary
            AwwasmOperands::I64Clz    => handle_op!(op_i64_clz, self, source, frame_idx),
            AwwasmOperands::I64Ctz    => handle_op!(op_i64_ctz, self, source, frame_idx),
            AwwasmOperands::I64Popcnt => handle_op!(op_i64_popcnt, self, source, frame_idx),
            // i64 arithmetic
            AwwasmOperands::I64Add  => handle_op!(op_i64_add, self, source, frame_idx),
            AwwasmOperands::I64Sub  => handle_op!(op_i64_sub, self, source, frame_idx),
            AwwasmOperands::I64Mul  => handle_op!(op_i64_mul, self, source, frame_idx),
            AwwasmOperands::I64DivS => handle_op!(op_i64_div_s, self, source, frame_idx),
            AwwasmOperands::I64DivU => handle_op!(op_i64_div_u, self, source, frame_idx),
            AwwasmOperands::I64RemS => handle_op!(op_i64_rem_s, self, source, frame_idx),
            AwwasmOperands::I64RemU => handle_op!(op_i64_rem_u, self, source, frame_idx),
            AwwasmOperands::I64And  => handle_op!(op_i64_and, self, source, frame_idx),
            AwwasmOperands::I64Or   => handle_op!(op_i64_or, self, source, frame_idx),
            AwwasmOperands::I64Xor  => handle_op!(op_i64_xor, self, source, frame_idx),
            AwwasmOperands::I64Shl  => handle_op!(op_i64_shl, self, source, frame_idx),
            AwwasmOperands::I64ShrS => handle_op!(op_i64_shr_s, self, source, frame_idx),
            AwwasmOperands::I64ShrU => handle_op!(op_i64_shr_u, self, source, frame_idx),
            AwwasmOperands::I64Rotl => handle_op!(op_i64_rotl, self, source, frame_idx),
            AwwasmOperands::I64Rotr => handle_op!(op_i64_rotr, self, source, frame_idx),

            // f32 comparisons
            AwwasmOperands::F32Eq => handle_op!(op_f32_eq, self, source, frame_idx),
            AwwasmOperands::F32Ne => handle_op!(op_f32_ne, self, source, frame_idx),
            AwwasmOperands::F32Lt => handle_op!(op_f32_lt, self, source, frame_idx),
            AwwasmOperands::F32Gt => handle_op!(op_f32_gt, self, source, frame_idx),
            AwwasmOperands::F32Le => handle_op!(op_f32_le, self, source, frame_idx),
            AwwasmOperands::F32Ge => handle_op!(op_f32_ge, self, source, frame_idx),
            // f64 comparisons
            AwwasmOperands::F64Eq => handle_op!(op_f64_eq, self, source, frame_idx),
            AwwasmOperands::F64Ne => handle_op!(op_f64_ne, self, source, frame_idx),
            AwwasmOperands::F64Lt => handle_op!(op_f64_lt, self, source, frame_idx),
            AwwasmOperands::F64Gt => handle_op!(op_f64_gt, self, source, frame_idx),
            AwwasmOperands::F64Le => handle_op!(op_f64_le, self, source, frame_idx),
            AwwasmOperands::F64Ge => handle_op!(op_f64_ge, self, source, frame_idx),
            // f32 arithmetic
            AwwasmOperands::F32Abs      => handle_op!(op_f32_abs, self, source, frame_idx),
            AwwasmOperands::F32Neg      => handle_op!(op_f32_neg, self, source, frame_idx),
            AwwasmOperands::F32Ceil     => handle_op!(op_f32_ceil, self, source, frame_idx),
            AwwasmOperands::F32Floor    => handle_op!(op_f32_floor, self, source, frame_idx),
            AwwasmOperands::F32Trunc    => handle_op!(op_f32_trunc, self, source, frame_idx),
            AwwasmOperands::F32Nearest  => handle_op!(op_f32_nearest, self, source, frame_idx),
            AwwasmOperands::F32Sqrt     => handle_op!(op_f32_sqrt, self, source, frame_idx),
            AwwasmOperands::F32Add      => handle_op!(op_f32_add, self, source, frame_idx),
            AwwasmOperands::F32Sub      => handle_op!(op_f32_sub, self, source, frame_idx),
            AwwasmOperands::F32Mul      => handle_op!(op_f32_mul, self, source, frame_idx),
            AwwasmOperands::F32Div      => handle_op!(op_f32_div, self, source, frame_idx),
            AwwasmOperands::F32Min      => handle_op!(op_f32_min, self, source, frame_idx),
            AwwasmOperands::F32Max      => handle_op!(op_f32_max, self, source, frame_idx),
            AwwasmOperands::F32Copysign => handle_op!(op_f32_copysign, self, source, frame_idx),
            // f64 arithmetic
            AwwasmOperands::F64Abs      => handle_op!(op_f64_abs, self, source, frame_idx),
            AwwasmOperands::F64Neg      => handle_op!(op_f64_neg, self, source, frame_idx),
            AwwasmOperands::F64Ceil     => handle_op!(op_f64_ceil, self, source, frame_idx),
            AwwasmOperands::F64Floor    => handle_op!(op_f64_floor, self, source, frame_idx),
            AwwasmOperands::F64Trunc    => handle_op!(op_f64_trunc, self, source, frame_idx),
            AwwasmOperands::F64Nearest  => handle_op!(op_f64_nearest, self, source, frame_idx),
            AwwasmOperands::F64Sqrt     => handle_op!(op_f64_sqrt, self, source, frame_idx),
            AwwasmOperands::F64Add      => handle_op!(op_f64_add, self, source, frame_idx),
            AwwasmOperands::F64Sub      => handle_op!(op_f64_sub, self, source, frame_idx),
            AwwasmOperands::F64Mul      => handle_op!(op_f64_mul, self, source, frame_idx),
            AwwasmOperands::F64Div      => handle_op!(op_f64_div, self, source, frame_idx),
            AwwasmOperands::F64Min      => handle_op!(op_f64_min, self, source, frame_idx),
            AwwasmOperands::F64Max      => handle_op!(op_f64_max, self, source, frame_idx),
            AwwasmOperands::F64Copysign => handle_op!(op_f64_copysign, self, source, frame_idx),

            // Type conversions
            AwwasmOperands::I32WrapI64      => handle_op!(op_i32_wrap_i64, self, source, frame_idx),
            AwwasmOperands::I32TruncF32S    => handle_op!(op_i32_trunc_f32_s, self, source, frame_idx),
            AwwasmOperands::I32TruncF32U    => handle_op!(op_i32_trunc_f32_u, self, source, frame_idx),
            AwwasmOperands::I32TruncF64S    => handle_op!(op_i32_trunc_f64_s, self, source, frame_idx),
            AwwasmOperands::I32TruncF64U    => handle_op!(op_i32_trunc_f64_u, self, source, frame_idx),
            AwwasmOperands::I64ExtendI32S   => handle_op!(op_i64_extend_i32_s, self, source, frame_idx),
            AwwasmOperands::I64ExtendI32U   => handle_op!(op_i64_extend_i32_u, self, source, frame_idx),
            AwwasmOperands::I64TruncF32S    => handle_op!(op_i64_trunc_f32_s, self, source, frame_idx),
            AwwasmOperands::I64TruncF32U    => handle_op!(op_i64_trunc_f32_u, self, source, frame_idx),
            AwwasmOperands::I64TruncF64S    => handle_op!(op_i64_trunc_f64_s, self, source, frame_idx),
            AwwasmOperands::I64TruncF64U    => handle_op!(op_i64_trunc_f64_u, self, source, frame_idx),
            AwwasmOperands::F32ConvertI32S  => handle_op!(op_f32_convert_i32_s, self, source, frame_idx),
            AwwasmOperands::F32ConvertI32U  => handle_op!(op_f32_convert_i32_u, self, source, frame_idx),
            AwwasmOperands::F32ConvertI64S  => handle_op!(op_f32_convert_i64_s, self, source, frame_idx),
            AwwasmOperands::F32ConvertI64U  => handle_op!(op_f32_convert_i64_u, self, source, frame_idx),
            AwwasmOperands::F32DemoteF64    => handle_op!(op_f32_demote_f64, self, source, frame_idx),
            AwwasmOperands::F64ConvertI32S  => handle_op!(op_f64_convert_i32_s, self, source, frame_idx),
            AwwasmOperands::F64ConvertI32U  => handle_op!(op_f64_convert_i32_u, self, source, frame_idx),
            AwwasmOperands::F64ConvertI64S  => handle_op!(op_f64_convert_i64_s, self, source, frame_idx),
            AwwasmOperands::F64ConvertI64U  => handle_op!(op_f64_convert_i64_u, self, source, frame_idx),
            AwwasmOperands::F64PromoteF32   => handle_op!(op_f64_promote_f32, self, source, frame_idx),
            AwwasmOperands::I32ReinterpretF32 => handle_op!(op_i32_reinterpret_f32, self, source, frame_idx),
            AwwasmOperands::I64ReinterpretF64 => handle_op!(op_i64_reinterpret_f64, self, source, frame_idx),
            AwwasmOperands::F32ReinterpretI32 => handle_op!(op_f32_reinterpret_i32, self, source, frame_idx),
            AwwasmOperands::F64ReinterpretI64 => handle_op!(op_f64_reinterpret_i64, self, source, frame_idx),

            // Sign-extension operators
            AwwasmOperands::I32Extend8S  => handle_op!(op_i32_extend8_s, self, source, frame_idx),
            AwwasmOperands::I32Extend16S => handle_op!(op_i32_extend16_s, self, source, frame_idx),
            AwwasmOperands::I64Extend8S  => handle_op!(op_i64_extend8_s, self, source, frame_idx),
            AwwasmOperands::I64Extend16S => handle_op!(op_i64_extend16_s, self, source, frame_idx),
            AwwasmOperands::I64Extend32S => handle_op!(op_i64_extend32_s, self, source, frame_idx),
            AwwasmOperands::Misc(op) => handle_op!(op_misc, self, source, frame_idx, op),
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
        let addr = Self::eff_addr(base, op.offset)?;
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
        let addr = Self::eff_addr(base, op.offset)?;
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
        let addr = Self::eff_addr(base, op.offset)?;
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
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let mem = self.store.mem_mut(mem_addr)?;
        mem.write_i64(addr, val)?;
        dispatch_next!(self, source, frame_idx)
    }

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_load<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_f32(addr)?;
        self.stack.push(AwwasmValue::F32(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_load<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_f64(addr)?;
        self.stack.push(AwwasmValue::F64(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_load8_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u8(addr)? as i8 as i32;
        self.stack.push(AwwasmValue::I32(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_load8_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u8(addr)? as i32;
        self.stack.push(AwwasmValue::I32(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_load16_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u16(addr)? as i16 as i32;
        self.stack.push(AwwasmValue::I32(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_load16_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u16(addr)? as i32;
        self.stack.push(AwwasmValue::I32(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_load8_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u8(addr)? as i8 as i64;
        self.stack.push(AwwasmValue::I64(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_load8_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u8(addr)? as i64;
        self.stack.push(AwwasmValue::I64(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_load16_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u16(addr)? as i16 as i64;
        self.stack.push(AwwasmValue::I64(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_load16_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u16(addr)? as i64;
        self.stack.push(AwwasmValue::I64(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_load32_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u32(addr)? as i32 as i64;
        self.stack.push(AwwasmValue::I64(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_load32_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        let val = self.store.mem(mem_addr)?.read_u32(addr)? as i64;
        self.stack.push(AwwasmValue::I64(val));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_store<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_f32()?;
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        self.store.mem_mut(mem_addr)?.write_f32(addr, val)?;
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_store<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_f64()?;
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        self.store.mem_mut(mem_addr)?.write_f64(addr, val)?;
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_store8<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_i32()?;
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        self.store.mem_mut(mem_addr)?.write_u8(addr, val as u8)?;
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_store16<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_i32()?;
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        self.store.mem_mut(mem_addr)?.write_u16(addr, val as u16)?;
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_store8<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_i64()?;
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        self.store.mem_mut(mem_addr)?.write_u8(addr, val as u8)?;
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_store16<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_i64()?;
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        self.store.mem_mut(mem_addr)?.write_u16(addr, val as u16)?;
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_store32<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MemArg) -> Result<ControlSignal, AwwasmRuntimeError> {
        let val = self.pop_i64()?;
        let base = self.pop_i32()? as u32;
        let addr = Self::eff_addr(base, op.offset)?;
        let mem_addr = self.resolve_mem(frame_idx, 0)?;
        self.store.mem_mut(mem_addr)?.write_u32(addr, val as u32)?;
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
    // Compute effective memory address, trapping if base + offset overflows u32.
    #[inline]
    fn eff_addr(base: u32, offset: u32) -> Result<u32, AwwasmRuntimeError> {
        (base as u64).checked_add(offset as u64)
            .filter(|&a| a <= u32::MAX as u64)
            .map(|a| a as u32)
            .ok_or_else(|| AwwasmRuntimeError::Trap(AwwasmTrap::MemoryOutOfBounds {
                offset: base, size: offset, memory_size: 0,
            }))
    }

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

    // ========================================================================
    // Additional pop helpers
    // ========================================================================

    #[inline]
    fn pop_f32(&mut self) -> Result<f32, AwwasmRuntimeError> {
        match self.pop()? {
            AwwasmValue::F32(v) => Ok(v),
            other => Err(AwwasmRuntimeError::TypeMismatch {
                expected: "f32".into(),
                got: format!("{:?}", other.value_type()),
            }),
        }
    }

    #[inline]
    fn pop_f64(&mut self) -> Result<f64, AwwasmRuntimeError> {
        match self.pop()? {
            AwwasmValue::F64(v) => Ok(v),
            other => Err(AwwasmRuntimeError::TypeMismatch {
                expected: "f64".into(),
                got: format!("{:?}", other.value_type()),
            }),
        }
    }

    // ========================================================================
    // i32 comparison handlers
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_lt_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(if a < b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_lt_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()? as u32; let a = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::I32(if a < b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_gt_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(if a > b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_gt_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()? as u32; let a = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::I32(if a > b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_le_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(if a <= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_le_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()? as u32; let a = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::I32(if a <= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_ge_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(if a >= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_ge_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()? as u32; let a = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::I32(if a >= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // i32 unary bit ops
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_clz<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(v.leading_zeros() as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_ctz<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(v.trailing_zeros() as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_popcnt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(v.count_ones() as i32));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // i32 div/rem
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_div_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        let result = a.checked_div(b).ok_or(AwwasmRuntimeError::Trap(AwwasmTrap::IntegerOverflow))?;
        self.stack.push(AwwasmValue::I32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_div_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()? as u32; let a = self.pop_i32()? as u32;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        self.stack.push(AwwasmValue::I32((a / b) as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_rem_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        // MIN % -1 = 0 (no overflow trap for rem)
        let result = if a == i32::MIN && b == -1 { 0 } else { a % b };
        self.stack.push(AwwasmValue::I32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_rem_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()? as u32; let a = self.pop_i32()? as u32;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        self.stack.push(AwwasmValue::I32((a % b) as i32));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // i32 bitwise/shift/rotate
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_and<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a & b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_or<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a | b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_xor<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a ^ b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_shl<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a.wrapping_shl((b as u32) & 31)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_shr_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a.wrapping_shr((b as u32) & 31)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_shr_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::I32(a.wrapping_shr((b as u32) & 31) as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_rotl<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a.rotate_left((b as u32) & 31)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_rotr<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i32()?; let a = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(a.rotate_right((b as u32) & 31)));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // i64 comparison handlers
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_eqz<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(if v == 0 { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_eq<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(if a == b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_ne<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(if a != b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_lt_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(if a < b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_lt_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()? as u64; let a = self.pop_i64()? as u64;
        self.stack.push(AwwasmValue::I32(if a < b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_gt_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(if a > b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_gt_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()? as u64; let a = self.pop_i64()? as u64;
        self.stack.push(AwwasmValue::I32(if a > b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_le_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(if a <= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_le_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()? as u64; let a = self.pop_i64()? as u64;
        self.stack.push(AwwasmValue::I32(if a <= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_ge_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(if a >= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_ge_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()? as u64; let a = self.pop_i64()? as u64;
        self.stack.push(AwwasmValue::I32(if a >= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // i64 unary + arithmetic
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_clz<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(v.leading_zeros() as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_ctz<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(v.trailing_zeros() as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_popcnt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(v.count_ones() as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_add<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a.wrapping_add(b)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_sub<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a.wrapping_sub(b)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_mul<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a.wrapping_mul(b)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_div_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        let result = a.checked_div(b).ok_or(AwwasmRuntimeError::Trap(AwwasmTrap::IntegerOverflow))?;
        self.stack.push(AwwasmValue::I64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_div_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()? as u64; let a = self.pop_i64()? as u64;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        self.stack.push(AwwasmValue::I64((a / b) as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_rem_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        let result = if a == i64::MIN && b == -1 { 0 } else { a % b };
        self.stack.push(AwwasmValue::I64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_rem_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()? as u64; let a = self.pop_i64()? as u64;
        if b == 0 { return Err(AwwasmTrap::DivisionByZero.into()); }
        self.stack.push(AwwasmValue::I64((a % b) as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_and<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a & b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_or<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a | b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_xor<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a ^ b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_shl<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a.wrapping_shl((b as u32) & 63)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_shr_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a.wrapping_shr((b as u32) & 63)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_shr_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()? as u64;
        self.stack.push(AwwasmValue::I64(a.wrapping_shr((b as u32) & 63) as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_rotl<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a.rotate_left((b as u32) & 63)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_rotr<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_i64()?; let a = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(a.rotate_right((b as u32) & 63)));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // f32 comparisons (return i32 0/1; Rust float comparisons yield false for NaN)
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_eq<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::I32(if a == b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_ne<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::I32(if a != b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_lt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::I32(if a < b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_gt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::I32(if a > b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_le<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::I32(if a <= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_ge<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::I32(if a >= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // f64 comparisons
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_eq<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::I32(if a == b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_ne<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::I32(if a != b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_lt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::I32(if a < b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_gt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::I32(if a > b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_le<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::I32(if a <= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_ge<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::I32(if a >= b { 1 } else { 0 }));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // f32 arithmetic
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_abs<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(v.abs()));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_neg<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(-v));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_ceil<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        let result = if v.is_nan() { f32::from_bits(v.to_bits() | 0x0040_0000) } else { v.ceil() };
        self.stack.push(AwwasmValue::F32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_floor<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        let result = if v.is_nan() { f32::from_bits(v.to_bits() | 0x0040_0000) } else { v.floor() };
        self.stack.push(AwwasmValue::F32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_trunc<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        let result = if v.is_nan() { f32::from_bits(v.to_bits() | 0x0040_0000) } else { v.trunc() };
        self.stack.push(AwwasmValue::F32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_nearest<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        let result = if v.is_nan() { f32::from_bits(v.to_bits() | 0x0040_0000) } else { v.round_ties_even() };
        self.stack.push(AwwasmValue::F32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_sqrt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(v.sqrt()));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_add<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(a + b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_sub<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(a - b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_mul<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(a * b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_div<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(a / b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_min<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        // WASM: NaN → canonical NaN; ±0.0 → -0.0 (OR bits to pick the negative zero)
        let result = if a.is_nan() || b.is_nan() { f32::from_bits(0x7FC0_0000) }
                     else if a == 0.0 && b == 0.0 { f32::from_bits(a.to_bits() | b.to_bits()) }
                     else { a.min(b) };
        self.stack.push(AwwasmValue::F32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_max<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        // WASM: NaN → canonical NaN; ±0.0 → +0.0 (AND bits to pick the positive zero)
        let result = if a.is_nan() || b.is_nan() { f32::from_bits(0x7FC0_0000) }
                     else if a == 0.0 && b == 0.0 { f32::from_bits(a.to_bits() & b.to_bits()) }
                     else { a.max(b) };
        self.stack.push(AwwasmValue::F32(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_copysign<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f32()?; let a = self.pop_f32()?;
        self.stack.push(AwwasmValue::F32(a.copysign(b)));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // f64 arithmetic
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_abs<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(v.abs()));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_neg<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(-v));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_ceil<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        let result = if v.is_nan() { f64::from_bits(v.to_bits() | 0x0008_0000_0000_0000) } else { v.ceil() };
        self.stack.push(AwwasmValue::F64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_floor<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        let result = if v.is_nan() { f64::from_bits(v.to_bits() | 0x0008_0000_0000_0000) } else { v.floor() };
        self.stack.push(AwwasmValue::F64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_trunc<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        let result = if v.is_nan() { f64::from_bits(v.to_bits() | 0x0008_0000_0000_0000) } else { v.trunc() };
        self.stack.push(AwwasmValue::F64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_nearest<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        let result = if v.is_nan() { f64::from_bits(v.to_bits() | 0x0008_0000_0000_0000) } else { v.round_ties_even() };
        self.stack.push(AwwasmValue::F64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_sqrt<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(v.sqrt()));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_add<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(a + b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_sub<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(a - b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_mul<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(a * b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_div<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(a / b));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_min<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        // WASM: NaN → canonical NaN; ±0.0 → -0.0 (OR bits to pick the negative zero)
        let result = if a.is_nan() || b.is_nan() { f64::from_bits(0x7FF8_0000_0000_0000) }
                     else if a == 0.0 && b == 0.0 { f64::from_bits(a.to_bits() | b.to_bits()) }
                     else { a.min(b) };
        self.stack.push(AwwasmValue::F64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_max<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        // WASM: NaN → canonical NaN; ±0.0 → +0.0 (AND bits to pick the positive zero)
        let result = if a.is_nan() || b.is_nan() { f64::from_bits(0x7FF8_0000_0000_0000) }
                     else if a == 0.0 && b == 0.0 { f64::from_bits(a.to_bits() & b.to_bits()) }
                     else { a.max(b) };
        self.stack.push(AwwasmValue::F64(result));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_copysign<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let b = self.pop_f64()?; let a = self.pop_f64()?;
        self.stack.push(AwwasmValue::F64(a.copysign(b)));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // Type conversion handlers
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_wrap_i64<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I32(v as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_trunc_f32_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        if v.is_nan() || v >= 2147483648.0_f32 || v < -2147483648.0_f32 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I32(v as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_trunc_f32_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        // Values in (-1, 0) truncate to 0 — valid; trap only if trunc result is negative (v <= -1)
        if v.is_nan() || v >= 4294967296.0_f32 || v <= -1.0_f32 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I32((v as u32) as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_trunc_f64_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        // Lower bound: -2147483648.9 truncates to -2147483648 (i32::MIN) which is valid.
        // Trap only when truncated result would be < i32::MIN, i.e. v <= -2147483649.0.
        if v.is_nan() || v >= 2147483648.0_f64 || v <= -2147483649.0_f64 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I32(v as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_trunc_f64_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        if v.is_nan() || v >= 4294967296.0_f64 || v <= -1.0_f64 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I32((v as u32) as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_extend_i32_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::I64(v as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_extend_i32_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::I64(v as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_trunc_f32_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        if v.is_nan() || v >= 9.223372036854776e18_f32 || v < -9.223372036854776e18_f32 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I64(v as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_trunc_f32_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        if v.is_nan() || v >= 1.8446744073709552e19_f32 || v <= -1.0_f32 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I64((v as u64) as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_trunc_f64_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        if v.is_nan() || v >= 9.223372036854776e18_f64 || v < -9.223372036854776e18_f64 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I64(v as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_trunc_f64_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        if v.is_nan() || v >= 1.8446744073709552e19_f64 || v <= -1.0_f64 {
            return Err(AwwasmTrap::InvalidConversionToInteger.into());
        }
        self.stack.push(AwwasmValue::I64((v as u64) as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_convert_i32_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::F32(v as f32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_convert_i32_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::F32(v as f32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_convert_i64_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::F32(v as f32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_convert_i64_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()? as u64;
        self.stack.push(AwwasmValue::F32(v as f32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_demote_f64<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        self.stack.push(AwwasmValue::F32(v as f32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_convert_i32_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::F64(v as f64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_convert_i32_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()? as u32;
        self.stack.push(AwwasmValue::F64(v as f64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_convert_i64_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::F64(v as f64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_convert_i64_u<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()? as u64;
        self.stack.push(AwwasmValue::F64(v as f64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_promote_f32<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        self.stack.push(AwwasmValue::F64(v as f64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_reinterpret_f32<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f32()?;
        self.stack.push(AwwasmValue::I32(v.to_bits() as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_reinterpret_f64<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_f64()?;
        self.stack.push(AwwasmValue::I64(v.to_bits() as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f32_reinterpret_i32<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::F32(f32::from_bits(v as u32)));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_f64_reinterpret_i64<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::F64(f64::from_bits(v as u64)));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // Sign-extension operators
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_extend8_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(v as i8 as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i32_extend16_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i32()?;
        self.stack.push(AwwasmValue::I32(v as i16 as i32));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_extend8_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(v as i8 as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_extend16_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(v as i16 as i64));
        dispatch_next!(self, source, frame_idx)
    }
    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_i64_extend32_s<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize) -> Result<ControlSignal, AwwasmRuntimeError> {
        let v = self.pop_i64()?;
        self.stack.push(AwwasmValue::I64(v as i32 as i64));
        dispatch_next!(self, source, frame_idx)
    }

    // ========================================================================
    // 0xFC prefix: saturating truncation and bulk memory ops
    // ========================================================================

    #[cfg_attr(feature = "tail_calls", inline(always))]
    fn op_misc<S: InstrSource<'a>>(&mut self, source: &S, frame_idx: usize, op: &awwasm_parser::components::instructions::MiscOperands) -> Result<ControlSignal, AwwasmRuntimeError> {
        match op.sub_op {
            0 => { // i32.trunc_sat_f32_s
                let v = self.pop_f32()?;
                let result = if v.is_nan() { 0 } else if v >= 2147483648.0_f32 { i32::MAX } else if v < -2147483648.0_f32 { i32::MIN } else { v as i32 };
                self.stack.push(AwwasmValue::I32(result));
            }
            1 => { // i32.trunc_sat_f32_u
                let v = self.pop_f32()?;
                let result = if v.is_nan() || v <= -1.0_f32 { 0u32 } else if v >= 4294967296.0_f32 { u32::MAX } else { v as u32 };
                self.stack.push(AwwasmValue::I32(result as i32));
            }
            2 => { // i32.trunc_sat_f64_s
                let v = self.pop_f64()?;
                let result = if v.is_nan() { 0 } else if v >= 2147483648.0_f64 { i32::MAX } else if v < -2147483648.0_f64 { i32::MIN } else { v as i32 };
                self.stack.push(AwwasmValue::I32(result));
            }
            3 => { // i32.trunc_sat_f64_u
                let v = self.pop_f64()?;
                let result = if v.is_nan() || v <= -1.0_f64 { 0u32 } else if v >= 4294967296.0_f64 { u32::MAX } else { v as u32 };
                self.stack.push(AwwasmValue::I32(result as i32));
            }
            4 => { // i64.trunc_sat_f32_s
                let v = self.pop_f32()?;
                let result = if v.is_nan() { 0i64 } else if v >= 9.223372036854776e18_f32 { i64::MAX } else if v < -9.223372036854776e18_f32 { i64::MIN } else { v as i64 };
                self.stack.push(AwwasmValue::I64(result));
            }
            5 => { // i64.trunc_sat_f32_u
                let v = self.pop_f32()?;
                let result = if v.is_nan() || v <= -1.0_f32 { 0u64 } else if v >= 1.8446744073709552e19_f32 { u64::MAX } else { v as u64 };
                self.stack.push(AwwasmValue::I64(result as i64));
            }
            6 => { // i64.trunc_sat_f64_s
                let v = self.pop_f64()?;
                let result = if v.is_nan() { 0i64 } else if v >= 9.223372036854776e18_f64 { i64::MAX } else if v < -9.223372036854776e18_f64 { i64::MIN } else { v as i64 };
                self.stack.push(AwwasmValue::I64(result));
            }
            7 => { // i64.trunc_sat_f64_u
                let v = self.pop_f64()?;
                let result = if v.is_nan() || v <= -1.0_f64 { 0u64 } else if v >= 1.8446744073709552e19_f64 { u64::MAX } else { v as u64 };
                self.stack.push(AwwasmValue::I64(result as i64));
            }
            other => return Err(AwwasmRuntimeError::InstructionParseError(format!("0xFC sub-op {other} not implemented"))),
        }
        dispatch_next!(self, source, frame_idx)
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
