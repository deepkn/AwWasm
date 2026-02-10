//! Function instance implementation.
//!
//! A function instance is a closure over module + code.
//! Supports lazy parsing of instruction bytes.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use core::marker::PhantomData;

use crate::values::AwwasmModuleAddr;

/// Function type signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwwasmFuncType {
    /// Parameter types.
    pub params: Vec<crate::values::AwwasmValueType>,
    /// Result types.
    pub results: Vec<crate::values::AwwasmValueType>,
}

impl AwwasmFuncType {
    /// Create a new function type.
    pub fn new(params: Vec<crate::values::AwwasmValueType>, results: Vec<crate::values::AwwasmValueType>) -> Self {
        Self { params, results }
    }
}

/// Lazy code representation for function bodies.
///
/// This enum enables lazy parsing: function bodies are stored as raw
/// bytes until they are actually executed, at which point they are
/// parsed into instructions.
#[derive(Debug, Clone)]
pub enum LazyResolvedCodeRef<'a> {
    /// Raw bytes, not yet parsed.
    /// Contains the locals + instruction sequence.
    Unparsed {
        /// Raw function body bytes (from parsed module).
        bytes: &'a [u8],
    },
    /// Parsed locals and code bytes ready for interpretation.
    /// Note: We keep code as bytes for the interpreter to use
    /// with InstructionIterator for streaming execution.
    Resolved {
        /// Local variable declarations (count, type).
        locals: Vec<AwwasmLocalDecl>,
        /// Code bytes (instruction sequence).
        code: &'a [u8],
    },
}

/// Local variable declaration in a function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwwasmLocalDecl {
    /// Number of locals of this type.
    pub count: u32,
    /// The value type.
    pub type_: crate::values::AwwasmValueType,
}

/// Function instance - runtime representation of a function.
///
/// This can be either a WebAssembly function (with code) or a
/// host function (provided by the embedder).
#[derive(Debug, Clone)]
pub enum AwwasmFuncInst<'a> {
    /// WebAssembly function defined in a module.
    Wasm(AwwasmWasmFuncInst<'a>),
    /// Host function provided by the embedder.
    Host(AwwasmHostFuncInst),
}

/// WebAssembly function instance.
#[derive(Debug, Clone)]
pub struct AwwasmWasmFuncInst<'a> {
    /// Index into the type section (for the function signature).
    pub type_idx: u32,
    /// Reference to the owning module instance.
    pub module: AwwasmModuleAddr,
    /// The function code (lazy-parsed).
    pub code: LazyResolvedCodeRef<'a>,
}

/// Host function instance.
///
/// The actual implementation is provided by the embedder.
/// We just store an identifier that the embedder can use to
/// look up the actual function.
#[derive(Debug, Clone)]
pub struct AwwasmHostFuncInst {
    /// Index into the type section (for the function signature).
    pub type_idx: u32,
    /// Embedder-defined host function identifier.
    pub host_func_id: u32,
}

impl<'a> AwwasmFuncInst<'a> {
    /// Create a new WebAssembly function instance.
    pub fn wasm(type_idx: u32, module: AwwasmModuleAddr, code_bytes: &'a [u8]) -> Self {
        AwwasmFuncInst::Wasm(AwwasmWasmFuncInst {
            type_idx,
            module,
            code: LazyResolvedCodeRef::Unparsed { bytes: code_bytes },
        })
    }

    /// Create a new host function instance.
    pub fn host(type_idx: u32, host_func_id: u32) -> Self {
        AwwasmFuncInst::Host(AwwasmHostFuncInst {
            type_idx,
            host_func_id,
        })
    }

    /// Get the type index of this function.
    pub fn type_idx(&self) -> u32 {
        match self {
            AwwasmFuncInst::Wasm(f) => f.type_idx,
            AwwasmFuncInst::Host(f) => f.type_idx,
        }
    }

    /// Check if this is a WebAssembly function.
    pub fn is_wasm(&self) -> bool {
        matches!(self, AwwasmFuncInst::Wasm(_))
    }

    /// Check if this is a host function.
    pub fn is_host(&self) -> bool {
        matches!(self, AwwasmFuncInst::Host(_))
    }
}

/// Element instance - runtime representation of an element segment.
#[derive(Debug, Clone)]
pub struct AwwasmElemInst<'a> {
    /// The element type.
    pub type_: crate::table::AwwasmElemType,
    /// The reference values.
    pub elem: Vec<Option<crate::values::AwwasmFuncAddr>>,
    /// Whether this segment has been dropped.
    pub dropped: bool,
    /// Preserve lifetime for future reference types.
    pub(crate) _phantom: PhantomData<&'a ()>,
}

impl<'a> AwwasmElemInst<'a> {
    /// Create a new element instance.
    pub fn new(type_: crate::table::AwwasmElemType, elem: Vec<Option<crate::values::AwwasmFuncAddr>>) -> Self {
        Self {
            type_,
            elem,
            dropped: false,
            _phantom: PhantomData,
        }
    }

    /// Drop this element segment (elem.drop instruction).
    pub fn drop_elem(&mut self) {
        self.elem.clear();
        self.dropped = true;
    }
}

/// Data instance - runtime representation of a data segment.
#[derive(Debug, Clone)]
pub struct AwwasmDataInst<'a> {
    /// The data bytes (zero-copy reference to parsed module).
    pub data: &'a [u8],
    /// Whether this segment has been dropped.
    pub dropped: bool,
}

impl<'a> AwwasmDataInst<'a> {
    /// Create a new data instance.
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            dropped: false,
        }
    }

    /// Drop this data segment (data.drop instruction).
    pub fn drop_data(&mut self) {
        // We can't actually free the bytes (they're borrowed),
        // but we mark it as dropped so further access fails.
        self.dropped = true;
    }

    /// Get the data bytes (if not dropped).
    pub fn bytes(&self) -> Option<&'a [u8]> {
        if self.dropped {
            None
        } else {
            Some(self.data)
        }
    }
}

