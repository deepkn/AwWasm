//! WebAssembly runtime values and addresses.
//!
//! This module defines the core value types and type-safe addresses
//! used throughout the runtime.

/// Runtime values that can appear on the stack or in globals.
///
/// Per the WebAssembly spec, values are either numbers or references.
/// Currently we support the four basic number types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AwwasmValue {
    /// 32-bit integer
    I32(i32),
    /// 64-bit integer
    I64(i64),
    /// 32-bit IEEE 754 floating point
    F32(f32),
    /// 64-bit IEEE 754 floating point
    F64(f64),
    // Future: V128 for SIMD
    // Future: FuncRef, ExternRef for reference types
}

impl AwwasmValue {
    /// Get the default value for a given value type.
    pub fn default_for_type(value_type: AwwasmValueType) -> Self {
        match value_type {
            AwwasmValueType::I32 => AwwasmValue::I32(0),
            AwwasmValueType::I64 => AwwasmValue::I64(0),
            AwwasmValueType::F32 => AwwasmValue::F32(0.0),
            AwwasmValueType::F64 => AwwasmValue::F64(0.0),
        }
    }

    /// Get the type of this value.
    pub fn value_type(&self) -> AwwasmValueType {
        match self {
            AwwasmValue::I32(_) => AwwasmValueType::I32,
            AwwasmValue::I64(_) => AwwasmValueType::I64,
            AwwasmValue::F32(_) => AwwasmValueType::F32,
            AwwasmValue::F64(_) => AwwasmValueType::F64,
        }
    }

    /// Try to get an i32 value.
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            AwwasmValue::I32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get an i64 value.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            AwwasmValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get an f32 value.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            AwwasmValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to get an f64 value.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AwwasmValue::F64(v) => Some(*v),
            _ => None,
        }
    }
}

/// Value types in WebAssembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AwwasmValueType {
    I32,
    I64,
    F32,
    F64,
}

// ============================================================================
// Type-safe addresses into Store components
// Using newtypes prevents mixing up different address types
// ============================================================================

/// Address of a function instance in the Store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AwwasmFuncAddr(pub u32);

/// Address of a table instance in the Store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AwwasmTableAddr(pub u32);

/// Address of a memory instance in the Store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AwwasmMemAddr(pub u32);

/// Address of a global instance in the Store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AwwasmGlobalAddr(pub u32);

/// Address of an element instance in the Store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AwwasmElemAddr(pub u32);

/// Address of a data instance in the Store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AwwasmDataAddr(pub u32);

/// Address of a module instance in the Store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AwwasmModuleAddr(pub u32);

/// External address - what can be imported/exported.
///
/// This represents the runtime address of an entity that can cross
/// module boundaries through imports and exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwwasmExternAddr {
    /// Function reference
    Func(AwwasmFuncAddr),
    /// Table reference
    Table(AwwasmTableAddr),
    /// Memory reference
    Mem(AwwasmMemAddr),
    /// Global reference
    Global(AwwasmGlobalAddr),
}

impl From<AwwasmFuncAddr> for AwwasmExternAddr {
    fn from(addr: AwwasmFuncAddr) -> Self {
        AwwasmExternAddr::Func(addr)
    }
}

impl From<AwwasmTableAddr> for AwwasmExternAddr {
    fn from(addr: AwwasmTableAddr) -> Self {
        AwwasmExternAddr::Table(addr)
    }
}

impl From<AwwasmMemAddr> for AwwasmExternAddr {
    fn from(addr: AwwasmMemAddr) -> Self {
        AwwasmExternAddr::Mem(addr)
    }
}

impl From<AwwasmGlobalAddr> for AwwasmExternAddr {
    fn from(addr: AwwasmGlobalAddr) -> Self {
        AwwasmExternAddr::Global(addr)
    }
}
