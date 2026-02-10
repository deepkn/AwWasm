//! Import resolution types for module instantiation.
//!
//! The embedder provides imports as `AwwasmImports`, which are matched
//! against a module's import section during `store_init()`.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::func::AwwasmFuncInst;
use crate::memory::AwwasmMemInst;
use crate::global::AwwasmGlobalInst;

/// Host-provided imports for module instantiation.
///
/// Imports are matched by (module, name) pairs against the module's
/// import section. The order does not matter.
#[derive(Debug)]
pub struct AwwasmImports<'a> {
    entries: Vec<AwwasmImportEntry<'a>>,
}

/// A single import entry keyed by (module, name).
#[derive(Debug)]
pub struct AwwasmImportEntry<'a> {
    /// Module name (e.g. "env").
    pub module: &'a [u8],
    /// Field name (e.g. "memory").
    pub name: &'a [u8],
    /// The provided value.
    pub value: AwwasmImportValue<'a>,
}

/// The value provided for an import.
#[derive(Debug)]
pub enum AwwasmImportValue<'a> {
    /// An imported function instance.
    Func(AwwasmFuncInst<'a>),
    /// An imported memory instance.
    Memory(AwwasmMemInst),
    /// An imported global instance.
    Global(AwwasmGlobalInst),
}

impl<'a> AwwasmImports<'a> {
    /// Create a new empty import set.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a function import.
    pub fn add_func(&mut self, module: &'a [u8], name: &'a [u8], func: AwwasmFuncInst<'a>) {
        self.entries.push(AwwasmImportEntry {
            module,
            name,
            value: AwwasmImportValue::Func(func),
        });
    }

    /// Add a memory import.
    pub fn add_memory(&mut self, module: &'a [u8], name: &'a [u8], mem: AwwasmMemInst) {
        self.entries.push(AwwasmImportEntry {
            module,
            name,
            value: AwwasmImportValue::Memory(mem),
        });
    }

    /// Add a global import.
    pub fn add_global(&mut self, module: &'a [u8], name: &'a [u8], global: AwwasmGlobalInst) {
        self.entries.push(AwwasmImportEntry {
            module,
            name,
            value: AwwasmImportValue::Global(global),
        });
    }

    /// Find an import by (module, name).
    pub fn find(&self, module: &[u8], name: &[u8]) -> Option<&AwwasmImportEntry<'a>> {
        self.entries.iter().find(|e| e.module == module && e.name == name)
    }

    /// Remove and return an import by (module, name).
    pub fn take(&mut self, module: &[u8], name: &[u8]) -> Option<AwwasmImportEntry<'a>> {
        let pos = self.entries.iter().position(|e| e.module == module && e.name == name)?;
        Some(self.entries.swap_remove(pos))
    }
}

impl<'a> Default for AwwasmImports<'a> {
    fn default() -> Self {
        Self::new()
    }
}
