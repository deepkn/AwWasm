//! Global instance implementation.
//!
//! A global instance is the runtime representation of a global variable.

use crate::values::AwwasmValue;

/// Global type - describes the mutability and value type of a global.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AwwasmGlobalType {
    /// Whether the global is mutable.
    pub mutable: bool,
    /// The value type of the global.
    pub value_type: crate::values::AwwasmValueType,
}

impl AwwasmGlobalType {
    /// Create a new immutable global type.
    pub fn immutable(value_type: crate::values::AwwasmValueType) -> Self {
        Self {
            mutable: false,
            value_type,
        }
    }

    /// Create a new mutable global type.
    pub fn mutable(value_type: crate::values::AwwasmValueType) -> Self {
        Self {
            mutable: true,
            value_type,
        }
    }
}

/// Global instance - runtime representation of a global variable.
#[derive(Debug, Clone)]
pub struct AwwasmGlobalInst {
    /// The global type (mutability + value type).
    pub type_: AwwasmGlobalType,
    /// The current value of the global.
    pub value: AwwasmValue,
}

impl AwwasmGlobalInst {
    /// Create a new global instance.
    pub fn new(type_: AwwasmGlobalType, value: AwwasmValue) -> Self {
        Self { type_, value }
    }

    /// Get the current value.
    #[inline]
    pub fn get(&self) -> AwwasmValue {
        self.value
    }

    /// Set the value (only if mutable).
    ///
    /// Returns an error if the global is immutable.
    #[inline]
    pub fn set(&mut self, value: AwwasmValue) -> Result<(), ()> {
        if !self.type_.mutable {
            return Err(());
        }
        self.value = value;
        Ok(())
    }

    /// Check if this global is mutable.
    #[inline]
    pub fn is_mutable(&self) -> bool {
        self.type_.mutable
    }
}

