//! Module instance and instantiation.
//!
//! A module instance is the runtime representation of an instantiated module.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::values::{AwwasmFuncAddr, AwwasmTableAddr, AwwasmMemAddr, AwwasmGlobalAddr, AwwasmElemAddr, AwwasmDataAddr, AwwasmExternAddr};

/// Export instance - runtime representation of an export.
#[derive(Debug, Clone)]
pub struct AwwasmExportInst<'a> {
    /// The export name (zero-copy from parsed module).
    pub name: &'a [u8],
    /// The external address.
    pub addr: AwwasmExternAddr,
}

impl<'a> AwwasmExportInst<'a> {
    /// Create a new export instance.
    pub fn new(name: &'a [u8], addr: AwwasmExternAddr) -> Self {
        Self { name, addr }
    }

    /// Get the export name as a string (if valid UTF-8).
    pub fn name_str(&self) -> Option<&str> {
        core::str::from_utf8(self.name).ok()
    }
}

/// Module instance - runtime representation of an instantiated module.
///
/// This holds all the addresses that map module-local indices to
/// Store addresses, plus the exports.
#[derive(Debug, Clone)]
pub struct AwwasmModuleInst<'a> {
    /// Function addresses (indexed by funcidx).
    pub funcaddrs: Vec<AwwasmFuncAddr>,
    /// Table addresses (indexed by tableidx).
    pub tableaddrs: Vec<AwwasmTableAddr>,
    /// Memory addresses (indexed by memidx).
    pub memaddrs: Vec<AwwasmMemAddr>,
    /// Global addresses (indexed by globalidx).
    pub globaladdrs: Vec<AwwasmGlobalAddr>,
    /// Element addresses (indexed by elemidx).
    pub elemaddrs: Vec<AwwasmElemAddr>,
    /// Data addresses (indexed by dataidx).
    pub dataaddrs: Vec<AwwasmDataAddr>,
    /// Exports.
    pub exports: Vec<AwwasmExportInst<'a>>,
    /// Start function (if any).
    pub start: Option<AwwasmFuncAddr>,
}

impl<'a> AwwasmModuleInst<'a> {
    /// Create a new empty module instance.
    pub fn new() -> Self {
        Self {
            funcaddrs: Vec::new(),
            tableaddrs: Vec::new(),
            memaddrs: Vec::new(),
            globaladdrs: Vec::new(),
            elemaddrs: Vec::new(),
            dataaddrs: Vec::new(),
            exports: Vec::new(),
            start: None,
        }
    }

    /// Get a function address by module-local index.
    pub fn func(&self, idx: u32) -> Option<AwwasmFuncAddr> {
        self.funcaddrs.get(idx as usize).copied()
    }

    /// Get a table address by module-local index.
    pub fn table(&self, idx: u32) -> Option<AwwasmTableAddr> {
        self.tableaddrs.get(idx as usize).copied()
    }

    /// Get a memory address by module-local index.
    pub fn mem(&self, idx: u32) -> Option<AwwasmMemAddr> {
        self.memaddrs.get(idx as usize).copied()
    }

    /// Get a global address by module-local index.
    pub fn global(&self, idx: u32) -> Option<AwwasmGlobalAddr> {
        self.globaladdrs.get(idx as usize).copied()
    }

    /// Get an element address by module-local index.
    pub fn elem(&self, idx: u32) -> Option<AwwasmElemAddr> {
        self.elemaddrs.get(idx as usize).copied()
    }

    /// Get a data address by module-local index.
    pub fn data(&self, idx: u32) -> Option<AwwasmDataAddr> {
        self.dataaddrs.get(idx as usize).copied()
    }

    /// Find an export by name.
    pub fn export(&self, name: &[u8]) -> Option<&AwwasmExportInst<'a>> {
        self.exports.iter().find(|e| e.name == name)
    }

    /// Find an export by string name.
    pub fn export_by_str(&self, name: &str) -> Option<&AwwasmExportInst<'a>> {
        self.export(name.as_bytes())
    }

    /// Get all function exports.
    pub fn func_exports(&self) -> impl Iterator<Item = (&'a [u8], AwwasmFuncAddr)> + '_ {
        self.exports.iter().filter_map(|e| {
            match e.addr {
                AwwasmExternAddr::Func(addr) => Some((e.name, addr)),
                _ => None,
            }
        })
    }

    /// Get all memory exports.
    pub fn mem_exports(&self) -> impl Iterator<Item = (&'a [u8], AwwasmMemAddr)> + '_ {
        self.exports.iter().filter_map(|e| {
            match e.addr {
                AwwasmExternAddr::Mem(addr) => Some((e.name, addr)),
                _ => None,
            }
        })
    }
}

impl<'a> Default for AwwasmModuleInst<'a> {
    fn default() -> Self {
        Self::new()
    }
}

