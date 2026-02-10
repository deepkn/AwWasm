//! Error types for the AwWasm runtime.

#[cfg(feature = "alloc")]
use alloc::string::String;

/// Errors that can occur during module instantiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwwasmInstantiationError {
    /// Import not found in provided imports
    MissingImport {
        module: String,
        name: String,
    },
    /// Import type mismatch
    ImportTypeMismatch {
        module: String,
        name: String,
        expected: String,
        got: String,
    },
    /// Memory allocation failed
    MemoryAllocationFailed {
        requested_pages: u32,
    },
    /// Data segment out of bounds
    DataSegmentOutOfBounds {
        segment_idx: u32,
        offset: u32,
        size: u32,
        memory_size: u32,
    },
    /// Element segment out of bounds
    ElementSegmentOutOfBounds {
        segment_idx: u32,
        offset: u32,
        size: u32,
        table_size: u32,
    },
    /// Start function trapped
    StartFunctionTrapped(AwwasmTrap),
    /// Unsupported type encountered during conversion
    UnsupportedType {
        description: String,
    },
    /// Invalid constant initializer expression
    InvalidConstExpr {
        description: String,
    },
    /// Function/code section count mismatch
    FuncCodeMismatch {
        func_count: u32,
        code_count: u32,
    },
}

/// Runtime trap - an unrecoverable error during execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwwasmTrap {
    /// Division by zero
    DivisionByZero,
    /// Integer overflow
    IntegerOverflow,
    /// Invalid conversion to integer
    InvalidConversionToInteger,
    /// Memory access out of bounds
    MemoryOutOfBounds {
        offset: u32,
        size: u32,
        memory_size: u32,
    },
    /// Table access out of bounds
    TableOutOfBounds {
        index: u32,
        table_size: u32,
    },
    /// Indirect call type mismatch
    IndirectCallTypeMismatch {
        expected_type: u32,
        actual_type: u32,
    },
    /// Indirect call to null reference
    IndirectCallToNull,
    /// Unreachable instruction executed
    Unreachable,
    /// Stack overflow
    StackOverflow,
    /// Call stack exhausted
    CallStackExhausted,
}

/// Errors that can occur during runtime execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AwwasmRuntimeError {
    /// A trap occurred
    Trap(AwwasmTrap),
    /// Invalid function address
    InvalidFuncAddr(u32),
    /// Invalid memory address
    InvalidMemAddr(u32),
    /// Invalid table address
    InvalidTableAddr(u32),
    /// Invalid global address
    InvalidGlobalAddr(u32),
    /// Attempted to execute a host function directly
    HostFunctionNotExecutable,
    /// Function has not been parsed yet
    FunctionNotParsed,
    /// Instruction parsing error
    InstructionParseError(String),
    /// Type mismatch on stack
    TypeMismatch {
        expected: String,
        got: String,
    },
    /// Global is immutable
    ImmutableGlobal(u32),
}

impl From<AwwasmTrap> for AwwasmRuntimeError {
    fn from(trap: AwwasmTrap) -> Self {
        AwwasmRuntimeError::Trap(trap)
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for AwwasmTrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AwwasmTrap::DivisionByZero => write!(f, "division by zero"),
            AwwasmTrap::IntegerOverflow => write!(f, "integer overflow"),
            AwwasmTrap::InvalidConversionToInteger => write!(f, "invalid conversion to integer"),
            AwwasmTrap::MemoryOutOfBounds { offset, size, memory_size } => {
                write!(f, "memory out of bounds: offset={}, size={}, memory_size={}", offset, size, memory_size)
            }
            AwwasmTrap::TableOutOfBounds { index, table_size } => {
                write!(f, "table out of bounds: index={}, table_size={}", index, table_size)
            }
            AwwasmTrap::IndirectCallTypeMismatch { expected_type, actual_type } => {
                write!(f, "indirect call type mismatch: expected={}, actual={}", expected_type, actual_type)
            }
            AwwasmTrap::IndirectCallToNull => write!(f, "indirect call to null"),
            AwwasmTrap::Unreachable => write!(f, "unreachable"),
            AwwasmTrap::StackOverflow => write!(f, "stack overflow"),
            AwwasmTrap::CallStackExhausted => write!(f, "call stack exhausted"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AwwasmTrap {}

#[cfg(feature = "std")]
impl std::fmt::Display for AwwasmRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AwwasmRuntimeError::Trap(trap) => write!(f, "trap: {}", trap),
            AwwasmRuntimeError::InvalidFuncAddr(addr) => write!(f, "invalid function address: {}", addr),
            AwwasmRuntimeError::InvalidMemAddr(addr) => write!(f, "invalid memory address: {}", addr),
            AwwasmRuntimeError::InvalidTableAddr(addr) => write!(f, "invalid table address: {}", addr),
            AwwasmRuntimeError::InvalidGlobalAddr(addr) => write!(f, "invalid global address: {}", addr),
            AwwasmRuntimeError::HostFunctionNotExecutable => write!(f, "cannot execute host function"),
            AwwasmRuntimeError::FunctionNotParsed => write!(f, "function not parsed"),
            AwwasmRuntimeError::InstructionParseError(msg) => write!(f, "instruction parse error: {}", msg),
            AwwasmRuntimeError::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {}, got {}", expected, got)
            }
            AwwasmRuntimeError::ImmutableGlobal(idx) => write!(f, "global {} is immutable", idx),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AwwasmRuntimeError {}
