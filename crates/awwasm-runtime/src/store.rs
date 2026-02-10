//! The Store - central runtime state for WebAssembly execution.
//!
//! The Store contains all runtime instances (functions, tables, memories,
//! globals, etc.) and provides allocation and access methods.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::values::{AwwasmFuncAddr, AwwasmTableAddr, AwwasmMemAddr, AwwasmGlobalAddr, AwwasmElemAddr, AwwasmDataAddr, AwwasmModuleAddr, AwwasmExternAddr};
use crate::func::{AwwasmFuncInst, AwwasmElemInst, AwwasmDataInst};
use crate::table::AwwasmTableInst;
use crate::memory::AwwasmMemInst;
use crate::global::AwwasmGlobalInst;
use crate::instance::{AwwasmModuleInst, AwwasmExportInst};
use crate::error::{AwwasmRuntimeError, AwwasmInstantiationError};
use crate::imports::{AwwasmImports, AwwasmImportValue};
use crate::type_convert;

use awwasm_parser::components::module::AwwasmModule;
use awwasm_parser::components::types::{AwwasmImportKind, AwwasmExportKind};

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

    /// Instantiate a parsed `AwwasmModule` into this Store.
    ///
    /// Entry point for the runtime. It:
    /// 1. Resolves imports and allocates imported instances
    /// 2. Allocates module-defined functions (lazy — bodies stay as raw `&[u8]`)
    /// 3. Allocates module-defined memories, globals
    /// 4. Allocates data segments (zero-copy `&'a [u8]` from parser)
    /// 5. Resolves exports
    /// 6. Initializes active data segments (copies bytes into linear memory)
    /// 7. Registers and returns the `AwwasmModuleAddr`
    pub fn store_init(
        &mut self,
        module: &AwwasmModule<'a>,
        imports: &mut AwwasmImports<'a>,
    ) -> Result<AwwasmModuleAddr, AwwasmInstantiationError> {
        let mut module_inst = AwwasmModuleInst::new();

        // Resolve imports
        if let Some(ref import_items) = module.imports {
            for import_item in import_items {
                let mod_name = import_item.module.bytes;
                let field_name = import_item.name.bytes;

                match import_item.kind {
                    AwwasmImportKind::Function => {
                        let entry = imports.take(mod_name, field_name).ok_or_else(|| {
                            AwwasmInstantiationError::MissingImport {
                                module: core::str::from_utf8(mod_name).unwrap_or("<invalid>").into(),
                                name: core::str::from_utf8(field_name).unwrap_or("<invalid>").into(),
                            }
                        })?;
                        match entry.value {
                            AwwasmImportValue::Func(func_inst) => {
                                let addr = self.alloc_func(func_inst);
                                module_inst.funcaddrs.push(addr);
                            }
                            _ => {
                                return Err(AwwasmInstantiationError::ImportTypeMismatch {
                                    module: core::str::from_utf8(mod_name).unwrap_or("<invalid>").into(),
                                    name: core::str::from_utf8(field_name).unwrap_or("<invalid>").into(),
                                    expected: "function".into(),
                                    got: "other".into(),
                                });
                            }
                        }
                    }
                    AwwasmImportKind::Memory => {
                        let entry = imports.take(mod_name, field_name).ok_or_else(|| {
                            AwwasmInstantiationError::MissingImport {
                                module: core::str::from_utf8(mod_name).unwrap_or("<invalid>").into(),
                                name: core::str::from_utf8(field_name).unwrap_or("<invalid>").into(),
                            }
                        })?;
                        match entry.value {
                            AwwasmImportValue::Memory(mem_inst) => {
                                let addr = self.alloc_mem(mem_inst);
                                module_inst.memaddrs.push(addr);
                            }
                            _ => {
                                return Err(AwwasmInstantiationError::ImportTypeMismatch {
                                    module: core::str::from_utf8(mod_name).unwrap_or("<invalid>").into(),
                                    name: core::str::from_utf8(field_name).unwrap_or("<invalid>").into(),
                                    expected: "memory".into(),
                                    got: "other".into(),
                                });
                            }
                        }
                    }
                    AwwasmImportKind::Global => {
                        let entry = imports.take(mod_name, field_name).ok_or_else(|| {
                            AwwasmInstantiationError::MissingImport {
                                module: core::str::from_utf8(mod_name).unwrap_or("<invalid>").into(),
                                name: core::str::from_utf8(field_name).unwrap_or("<invalid>").into(),
                            }
                        })?;
                        match entry.value {
                            AwwasmImportValue::Global(global_inst) => {
                                let addr = self.alloc_global(global_inst);
                                module_inst.globaladdrs.push(addr);
                            }
                            _ => {
                                return Err(AwwasmInstantiationError::ImportTypeMismatch {
                                    module: core::str::from_utf8(mod_name).unwrap_or("<invalid>").into(),
                                    name: core::str::from_utf8(field_name).unwrap_or("<invalid>").into(),
                                    expected: "global".into(),
                                    got: "other".into(),
                                });
                            }
                        }
                    }
                    // Table imports not yet supported
                    AwwasmImportKind::Table => {
                        return Err(AwwasmInstantiationError::UnsupportedType {
                            description: "table imports not yet supported".into(),
                        });
                    }
                }
            }
        }

        // Allocate module-defined functions
        let func_items = module.funcs.as_deref().unwrap_or(&[]);
        let code_items = module.code.as_deref().unwrap_or(&[]);

        if !func_items.is_empty() && func_items.len() != code_items.len() {
            return Err(AwwasmInstantiationError::FuncCodeMismatch {
                func_count: func_items.len() as u32,
                code_count: code_items.len() as u32,
            });
        }

        // Pre-compute the module address
        let pending_module_addr = AwwasmModuleAddr(self.modules.len() as u32);

        for code_item in code_items {
            // Store the raw func_body bytes — zero-copy from parser.
            // Resolution happens later (on-demand or via async batch).
            let func = AwwasmFuncInst::wasm(0, pending_module_addr, code_item.func_body);
            let addr = self.alloc_func(func);
            module_inst.funcaddrs.push(addr);
        }

        // Allocate module-defined memories
        if let Some(ref mem_items) = module.memories {
            for mem_item in mem_items {
                let mem_type = type_convert::memory_params_to_type(&mem_item.limits);
                let mem = AwwasmMemInst::new(mem_type);
                let addr = self.alloc_mem(mem);
                module_inst.memaddrs.push(addr);
            }
        }

        // Allocate data segments - zero-copy from parser
        if let Some(ref data_items) = module.data {
            for data_item in data_items {
                let data_inst = AwwasmDataInst::new(data_item.data_bytes);
                let addr = self.alloc_data(data_inst);
                module_inst.dataaddrs.push(addr);
            }
        }

        // Resolve exports
        if let Some(ref export_items) = module.exports {
            for export_item in export_items {
                let addr = match export_item.kind {
                    AwwasmExportKind::Function => {
                        let func_addr = module_inst.funcaddrs.get(export_item.index as usize)
                            .copied()
                            .ok_or_else(|| AwwasmInstantiationError::MissingImport {
                                module: "self".into(),
                                name: core::str::from_utf8(export_item.name.bytes).unwrap_or("<invalid>").into(),
                            })?;
                        AwwasmExternAddr::Func(func_addr)
                    }
                    AwwasmExportKind::Memory => {
                        let mem_addr = module_inst.memaddrs.get(export_item.index as usize)
                            .copied()
                            .ok_or_else(|| AwwasmInstantiationError::MissingImport {
                                module: "self".into(),
                                name: core::str::from_utf8(export_item.name.bytes).unwrap_or("<invalid>").into(),
                            })?;
                        AwwasmExternAddr::Mem(mem_addr)
                    }
                    AwwasmExportKind::Table => {
                        let table_addr = module_inst.tableaddrs.get(export_item.index as usize)
                            .copied()
                            .ok_or_else(|| AwwasmInstantiationError::MissingImport {
                                module: "self".into(),
                                name: core::str::from_utf8(export_item.name.bytes).unwrap_or("<invalid>").into(),
                            })?;
                        AwwasmExternAddr::Table(table_addr)
                    }
                    AwwasmExportKind::Global => {
                        let global_addr = module_inst.globaladdrs.get(export_item.index as usize)
                            .copied()
                            .ok_or_else(|| AwwasmInstantiationError::MissingImport {
                                module: "self".into(),
                                name: core::str::from_utf8(export_item.name.bytes).unwrap_or("<invalid>").into(),
                            })?;
                        AwwasmExternAddr::Global(global_addr)
                    }
                };
                module_inst.exports.push(AwwasmExportInst::new(export_item.name.bytes, addr));
            }
        }

        // Initialize active data segments
        // Active segments (flags 0x00 or 0x02) copy data into memory.
        // This is the one necessary memcpy — the source data_bytes is a
        // zero-copy &'a [u8] borrow from the parser, but linear memory
        // is mutable Vec<u8>, so the copy is required by the wasm spec.
        if let Some(ref data_items) = module.data {
            for (seg_idx, data_item) in data_items.iter().enumerate() {
                let flags = data_item.header.flags;

                // flags 0x00 = active, implicit memidx 0
                // flags 0x02 = active, explicit memidx
                // flags 0x01 = passive (skip)
                if flags == 0x01 {
                    continue;
                }

                let memidx = if flags == 0x02 {
                    data_item.header.memidx.unwrap_or(0) as usize
                } else {
                    0
                };

                let offset_expr = data_item.header.offset.as_ref().ok_or_else(|| {
                    AwwasmInstantiationError::InvalidConstExpr {
                        description: format!("data segment {} missing offset expression", seg_idx),
                    }
                })?;

                let offset = type_convert::eval_const_expr(offset_expr.code)? as usize;
                let data_bytes = data_item.data_bytes;

                // Resolve memidx through module instance to Store address
                let mem_addr = module_inst.memaddrs.get(memidx).copied().ok_or_else(|| {
                    AwwasmInstantiationError::DataSegmentOutOfBounds {
                        segment_idx: seg_idx as u32,
                        offset: offset as u32,
                        size: data_bytes.len() as u32,
                        memory_size: 0,
                    }
                })?;

                let mem = self.mems.get_mut(mem_addr.0 as usize).ok_or_else(|| {
                    AwwasmInstantiationError::DataSegmentOutOfBounds {
                        segment_idx: seg_idx as u32,
                        offset: offset as u32,
                        size: data_bytes.len() as u32,
                        memory_size: 0,
                    }
                })?;

                let mem_size = mem.data.len();
                if offset + data_bytes.len() > mem_size {
                    return Err(AwwasmInstantiationError::DataSegmentOutOfBounds {
                        segment_idx: seg_idx as u32,
                        offset: offset as u32,
                        size: data_bytes.len() as u32,
                        memory_size: mem_size as u32,
                    });
                }

                // The actual memcpy — unavoidable per wasm spec
                mem.data[offset..offset + data_bytes.len()].copy_from_slice(data_bytes);
            }
        }

        // Register module instance
        let addr = self.register_module(module_inst);

        Ok(addr)
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

