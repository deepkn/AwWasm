//! The Store - central runtime state for WebAssembly execution.
//!
//! The Store contains all runtime instances (functions, tables, memories,
//! globals, etc.) and provides allocation and access methods.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::values::{AwwasmFuncAddr, AwwasmTableAddr, AwwasmMemAddr, AwwasmGlobalAddr, AwwasmElemAddr, AwwasmDataAddr, AwwasmModuleAddr};
use crate::func::{AwwasmFuncInst, AwwasmElemInst, AwwasmDataInst};
use crate::table::AwwasmTableInst;
use crate::memory::AwwasmMemInst;
use crate::global::AwwasmGlobalInst;
use crate::instance::AwwasmModuleInst;
use crate::error::AwwasmRuntimeError;

/// The Store - global runtime state for WebAssembly.
///
/// Per the WebAssembly spec, the Store represents all global state that can
/// be manipulated by WebAssembly programs. It consists of runtime instances
/// of functions, tables, memories, globals, element segments, and data segments.
///
/// Multiple modules can share a Store, enabling cross-module calls and
/// shared memories/tables.
#[derive(Debug)]
pub struct AwwasmStore<'a> {
    /// Function instances.
    pub funcs: Vec<AwwasmFuncInst<'a>>,
    /// Table instances.
    pub tables: Vec<AwwasmTableInst>,
    /// Memory instances.
    pub mems: Vec<AwwasmMemInst>,
    /// Global instances.
    pub globals: Vec<AwwasmGlobalInst>,
    /// Element instances.
    pub elems: Vec<AwwasmElemInst<'a>>,
    /// Data instances.
    pub datas: Vec<AwwasmDataInst<'a>>,
    /// Module instances.
    pub modules: Vec<AwwasmModuleInst<'a>>,
}

impl<'a> AwwasmStore<'a> {
    /// Create a new empty Store.
    pub fn new() -> Self {
        Self {
            funcs: Vec::new(),
            tables: Vec::new(),
            mems: Vec::new(),
            globals: Vec::new(),
            elems: Vec::new(),
            datas: Vec::new(),
            modules: Vec::new(),
        }
    }

    // ========================================================================
    // Allocation methods
    // ========================================================================

    /// Allocate a function instance in the Store.
    pub fn alloc_func(&mut self, func: AwwasmFuncInst<'a>) -> AwwasmFuncAddr {
        let addr = AwwasmFuncAddr(self.funcs.len() as u32);
        self.funcs.push(func);
        addr
    }

    /// Allocate a table instance in the Store.
    pub fn alloc_table(&mut self, table: AwwasmTableInst) -> AwwasmTableAddr {
        let addr = AwwasmTableAddr(self.tables.len() as u32);
        self.tables.push(table);
        addr
    }

    /// Allocate a memory instance in the Store.
    pub fn alloc_mem(&mut self, mem: AwwasmMemInst) -> AwwasmMemAddr {
        let addr = AwwasmMemAddr(self.mems.len() as u32);
        self.mems.push(mem);
        addr
    }

    /// Allocate a global instance in the Store.
    pub fn alloc_global(&mut self, global: AwwasmGlobalInst) -> AwwasmGlobalAddr {
        let addr = AwwasmGlobalAddr(self.globals.len() as u32);
        self.globals.push(global);
        addr
    }

    /// Allocate an element instance in the Store.
    pub fn alloc_elem(&mut self, elem: AwwasmElemInst<'a>) -> AwwasmElemAddr {
        let addr = AwwasmElemAddr(self.elems.len() as u32);
        self.elems.push(elem);
        addr
    }

    /// Allocate a data instance in the Store.
    pub fn alloc_data(&mut self, data: AwwasmDataInst<'a>) -> AwwasmDataAddr {
        let addr = AwwasmDataAddr(self.datas.len() as u32);
        self.datas.push(data);
        addr
    }

    /// Register a module instance in the Store.
    pub fn register_module(&mut self, module: AwwasmModuleInst<'a>) -> AwwasmModuleAddr {
        let addr = AwwasmModuleAddr(self.modules.len() as u32);
        self.modules.push(module);
        addr
    }

    // ========================================================================
    // Access methods
    // ========================================================================

    /// Get a function instance by address.
    pub fn func(&self, addr: AwwasmFuncAddr) -> Result<&AwwasmFuncInst<'a>, AwwasmRuntimeError> {
        self.funcs
            .get(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidFuncAddr(addr.0))
    }

    /// Get a mutable function instance by address.
    pub fn func_mut(&mut self, addr: AwwasmFuncAddr) -> Result<&mut AwwasmFuncInst<'a>, AwwasmRuntimeError> {
        self.funcs
            .get_mut(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidFuncAddr(addr.0))
    }

    /// Get a table instance by address.
    pub fn table(&self, addr: AwwasmTableAddr) -> Result<&AwwasmTableInst, AwwasmRuntimeError> {
        self.tables
            .get(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidTableAddr(addr.0))
    }

    /// Get a mutable table instance by address.
    pub fn table_mut(&mut self, addr: AwwasmTableAddr) -> Result<&mut AwwasmTableInst, AwwasmRuntimeError> {
        self.tables
            .get_mut(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidTableAddr(addr.0))
    }

    /// Get a memory instance by address.
    pub fn mem(&self, addr: AwwasmMemAddr) -> Result<&AwwasmMemInst, AwwasmRuntimeError> {
        self.mems
            .get(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidMemAddr(addr.0))
    }

    /// Get a mutable memory instance by address.
    pub fn mem_mut(&mut self, addr: AwwasmMemAddr) -> Result<&mut AwwasmMemInst, AwwasmRuntimeError> {
        self.mems
            .get_mut(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidMemAddr(addr.0))
    }

    /// Get a global instance by address.
    pub fn global(&self, addr: AwwasmGlobalAddr) -> Result<&AwwasmGlobalInst, AwwasmRuntimeError> {
        self.globals
            .get(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidGlobalAddr(addr.0))
    }

    /// Get a mutable global instance by address.
    pub fn global_mut(&mut self, addr: AwwasmGlobalAddr) -> Result<&mut AwwasmGlobalInst, AwwasmRuntimeError> {
        self.globals
            .get_mut(addr.0 as usize)
            .ok_or(AwwasmRuntimeError::InvalidGlobalAddr(addr.0))
    }

    /// Get an element instance by address.
    pub fn elem(&self, addr: AwwasmElemAddr) -> Option<&AwwasmElemInst<'a>> {
        self.elems.get(addr.0 as usize)
    }

    /// Get a mutable element instance by address.
    pub fn elem_mut(&mut self, addr: AwwasmElemAddr) -> Option<&mut AwwasmElemInst<'a>> {
        self.elems.get_mut(addr.0 as usize)
    }

    /// Get a data instance by address.
    pub fn data(&self, addr: AwwasmDataAddr) -> Option<&AwwasmDataInst<'a>> {
        self.datas.get(addr.0 as usize)
    }

    /// Get a mutable data instance by address.
    pub fn data_mut(&mut self, addr: AwwasmDataAddr) -> Option<&mut AwwasmDataInst<'a>> {
        self.datas.get_mut(addr.0 as usize)
    }

    /// Get a module instance by address.
    pub fn module(&self, addr: AwwasmModuleAddr) -> Option<&AwwasmModuleInst<'a>> {
        self.modules.get(addr.0 as usize)
    }

    /// Get a mutable module instance by address.
    pub fn module_mut(&mut self, addr: AwwasmModuleAddr) -> Option<&mut AwwasmModuleInst<'a>> {
        self.modules.get_mut(addr.0 as usize)
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    /// Get the number of function instances.
    pub fn func_count(&self) -> usize {
        self.funcs.len()
    }

    /// Get the number of table instances.
    pub fn table_count(&self) -> usize {
        self.tables.len()
    }

    /// Get the number of memory instances.
    pub fn mem_count(&self) -> usize {
        self.mems.len()
    }

    /// Get the number of global instances.
    pub fn global_count(&self) -> usize {
        self.globals.len()
    }

    /// Get the number of module instances.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }
}

impl<'a> Default for AwwasmStore<'a> {
    fn default() -> Self {
        Self::new()
    }
}

