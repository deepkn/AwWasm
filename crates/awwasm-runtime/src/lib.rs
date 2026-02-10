//! AwWasm Runtime - A safe, minimal WebAssembly runtime
//!
//! This crate provides the execution environment for WebAssembly modules
//! parsed by `awwasm-parser`.
//!
//! # Features
//!
//! - `std` (default): Enable standard library support
//! - `alloc`: Enable heap allocation without full std
//! - `parallel`: Enable Rayon-based parallel parsing (future)

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod values;
pub mod memory;
pub mod global;
pub mod func;
pub mod table;
pub mod store;
pub mod instance;
pub mod type_convert;
pub mod imports;

// Re-export key types
pub use error::{AwwasmRuntimeError, AwwasmInstantiationError, AwwasmTrap};
pub use values::{AwwasmValue, AwwasmFuncAddr, AwwasmTableAddr, AwwasmMemAddr, AwwasmGlobalAddr, AwwasmElemAddr, AwwasmDataAddr, AwwasmExternAddr};
pub use store::AwwasmStore;
pub use instance::AwwasmModuleInst;
pub use imports::AwwasmImports;

#[cfg(test)]
mod tests {
    use super::*;
    use memory::{AwwasmMemInst, AwwasmMemoryType};
    use table::{AwwasmTableInst, AwwasmTableType};
    use global::{AwwasmGlobalInst, AwwasmGlobalType};
    use func::{AwwasmFuncInst, AwwasmDataInst};
    use values::{AwwasmValueType, AwwasmModuleAddr};

    #[test]
    fn test_store_creation() {
        let store: AwwasmStore = AwwasmStore::new();
        assert_eq!(store.func_count(), 0);
        assert_eq!(store.mem_count(), 0);
        assert_eq!(store.table_count(), 0);
        assert_eq!(store.global_count(), 0);
    }

    #[test]
    fn test_memory_allocation_and_access() {
        let mut store: AwwasmStore = AwwasmStore::new();
        
        // Allocate a memory with 1 page minimum
        let mem = AwwasmMemInst::new(AwwasmMemoryType::new(1, Some(2)));
        let addr = store.alloc_mem(mem);
        
        assert_eq!(store.mem_count(), 1);
        
        // Write and read back
        let mem = store.mem_mut(addr).unwrap();
        mem.write_i32(0, 42).unwrap();
        assert_eq!(mem.read_i32(0).unwrap(), 42);
        
        // Test grow
        let old_size = mem.grow(1).unwrap();
        assert_eq!(old_size, 1);
        assert_eq!(mem.size_pages(), 2);
        
        // Grow beyond max should fail
        assert!(mem.grow(1).is_none());
    }

    #[test]
    fn test_memory_bounds_checking() {
        let mem = AwwasmMemInst::new(AwwasmMemoryType::new(1, None));
        
        // Valid read at the end of page
        assert!(mem.read(65532, 4).is_ok());
        
        // Out of bounds read
        assert!(mem.read(65533, 4).is_err());
        assert!(mem.read(65536, 1).is_err());
    }

    #[test]
    fn test_table_allocation_and_access() {
        let mut store: AwwasmStore = AwwasmStore::new();
        
        let table = AwwasmTableInst::new(AwwasmTableType::funcref(4, Some(8)));
        let addr = store.alloc_table(table);
        
        let table = store.table_mut(addr).unwrap();
        assert_eq!(table.size(), 4);
        
        // Set and get
        table.set(0, Some(AwwasmFuncAddr(42))).unwrap();
        assert_eq!(table.get(0).unwrap(), Some(AwwasmFuncAddr(42)));
        
        // Null by default
        assert_eq!(table.get(1).unwrap(), None);
        
        // Grow
        let old = table.grow(2, None).unwrap();
        assert_eq!(old, 4);
        assert_eq!(table.size(), 6);
        
        // Bounds checking
        assert!(table.get(10).is_err());
    }

    #[test]
    fn test_global_allocation() {
        let mut store: AwwasmStore = AwwasmStore::new();
        
        // Mutable global
        let global = AwwasmGlobalInst::new(
            AwwasmGlobalType::mutable(AwwasmValueType::I32),
            AwwasmValue::I32(100),
        );
        let addr = store.alloc_global(global);
        
        let global = store.global_mut(addr).unwrap();
        assert_eq!(global.get(), AwwasmValue::I32(100));
        
        global.set(AwwasmValue::I32(200)).unwrap();
        assert_eq!(global.get(), AwwasmValue::I32(200));
    }

    #[test]
    fn test_immutable_global() {
        let mut global = AwwasmGlobalInst::new(
            AwwasmGlobalType::immutable(AwwasmValueType::I64),
            AwwasmValue::I64(42),
        );
        
        assert_eq!(global.get(), AwwasmValue::I64(42));
        assert!(!global.is_mutable());
        assert!(global.set(AwwasmValue::I64(0)).is_err());
    }

    #[test]
    fn test_function_allocation() {
        let mut store: AwwasmStore = AwwasmStore::new();
        
        // Create a wasm function
        let code: &[u8] = &[0x00, 0x0b]; // empty function body
        let func = AwwasmFuncInst::wasm(0, AwwasmModuleAddr(0), code);
        let addr = store.alloc_func(func);
        
        assert_eq!(store.func_count(), 1);
        
        let func = store.func(addr).unwrap();
        assert!(func.is_wasm());
        assert_eq!(func.type_idx(), 0);
    }

    #[test]
    fn test_data_instance() {
        let data_bytes: &[u8] = b"Hello, WebAssembly!";
        let mut data = AwwasmDataInst::new(data_bytes);
        
        assert_eq!(data.bytes(), Some(data_bytes));
        assert!(!data.dropped);
        
        data.drop_data();
        assert!(data.dropped);
        assert_eq!(data.bytes(), None);
    }

    #[test]
    fn test_module_instance() {
        let mut module = AwwasmModuleInst::new();
        
        module.funcaddrs.push(AwwasmFuncAddr(0));
        module.funcaddrs.push(AwwasmFuncAddr(1));
        module.memaddrs.push(AwwasmMemAddr(0));
        
        assert_eq!(module.func(0), Some(AwwasmFuncAddr(0)));
        assert_eq!(module.func(1), Some(AwwasmFuncAddr(1)));
        assert_eq!(module.func(2), None);
        assert_eq!(module.mem(0), Some(AwwasmMemAddr(0)));
    }

    #[test]
    fn test_value_types() {
        assert_eq!(AwwasmValue::I32(42).value_type(), AwwasmValueType::I32);
        assert_eq!(AwwasmValue::I64(100).value_type(), AwwasmValueType::I64);
        assert_eq!(AwwasmValue::F32(3.14).value_type(), AwwasmValueType::F32);
        assert_eq!(AwwasmValue::F64(2.718).value_type(), AwwasmValueType::F64);
        
        assert_eq!(AwwasmValue::I32(42).as_i32(), Some(42));
        assert_eq!(AwwasmValue::I32(42).as_i64(), None);
        
        assert_eq!(AwwasmValue::default_for_type(AwwasmValueType::I32), AwwasmValue::I32(0));
        assert_eq!(AwwasmValue::default_for_type(AwwasmValueType::F64), AwwasmValue::F64(0.0));
    }

    // store_init() integration tests
    use awwasm_parser::components::module::AwwasmModule;
    use crate::imports::AwwasmImports;
    use crate::func::LazyResolvedCodeRef;

    #[test]
    fn test_instantiate_minimal_module() {
        let wasm = wat::parse_str("(module)").unwrap();
        let mut module = AwwasmModule::new(&wasm).unwrap();
        // Empty module has no sections — skip resolve_all_sections
        // (parser panics with unwrap on None sections)
        if module.sections.is_some() {
            module.resolve_all_sections().unwrap();
        }

        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();
        let addr = store.store_init(&module, &mut imports).unwrap();

        assert_eq!(addr.0, 0);
        assert_eq!(store.module_count(), 1);
        let inst = store.module(addr).unwrap();
        assert_eq!(inst.funcaddrs.len(), 0);
        assert_eq!(inst.memaddrs.len(), 0);
    }

    #[test]
    fn test_instantiate_with_memory() {
        let wasm = wat::parse_str("(module (memory 1 4))").unwrap();
        let mut module = AwwasmModule::new(&wasm).unwrap();
        module.resolve_all_sections().unwrap();

        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();
        let addr = store.store_init(&module, &mut imports).unwrap();

        let inst = store.module(addr).unwrap();
        assert_eq!(inst.memaddrs.len(), 1);
        let mem = store.mem(inst.memaddrs[0]).unwrap();
        assert_eq!(mem.data.len(), 65536);
    }

    #[test]
    fn test_instantiate_with_function() {
        let wasm = wat::parse_str(
            "(module (func (result i32) (i32.const 42)))"
        ).unwrap();
        let mut module = AwwasmModule::new(&wasm).unwrap();
        module.resolve_all_sections().unwrap();

        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();
        let addr = store.store_init(&module, &mut imports).unwrap();

        let inst = store.module(addr).unwrap();
        assert_eq!(inst.funcaddrs.len(), 1);

        let func = store.func(inst.funcaddrs[0]).unwrap();
        match func {
            crate::func::AwwasmFuncInst::Wasm(wasm_func) => {
                assert!(matches!(wasm_func.code, LazyResolvedCodeRef::Unparsed { .. }));
            }
            _ => panic!("expected wasm function"),
        }
    }

    #[test]
    fn test_instantiate_with_data_segment() {
        let wasm = wat::parse_str(r#"
            (module
                (memory 1)
                (data (i32.const 16) "hello")
            )
        "#).unwrap();
        let mut module = AwwasmModule::new(&wasm).unwrap();
        module.resolve_all_sections().unwrap();

        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();
        let addr = store.store_init(&module, &mut imports).unwrap();

        let inst = store.module(addr).unwrap();
        let mem = store.mem(inst.memaddrs[0]).unwrap();

        assert_eq!(&mem.data[16..21], b"hello");
        assert_eq!(mem.data[15], 0);
        assert_eq!(mem.data[21], 0);
    }

    #[test]
    fn test_instantiate_with_exports() {
        let wasm = wat::parse_str(r#"
            (module
                (memory (export "memory") 1)
                (func (export "add") (result i32) (i32.const 0))
            )
        "#).unwrap();
        let mut module = AwwasmModule::new(&wasm).unwrap();
        module.resolve_all_sections().unwrap();

        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();
        let addr = store.store_init(&module, &mut imports).unwrap();

        let inst = store.module(addr).unwrap();
        assert_eq!(inst.exports.len(), 2);

        let mem_export = inst.exports.iter().find(|e| e.name == b"memory").unwrap();
        assert!(matches!(mem_export.addr, AwwasmExternAddr::Mem(_)));

        let func_export = inst.exports.iter().find(|e| e.name == b"add").unwrap();
        assert!(matches!(func_export.addr, AwwasmExternAddr::Func(_)));
    }

    #[test]
    fn test_instantiate_with_imports() {
        let wasm = wat::parse_str(r#"
            (module
                (import "env" "memory" (memory 1))
            )
        "#).unwrap();
        let mut module = AwwasmModule::new(&wasm).unwrap();
        module.resolve_all_sections().unwrap();

        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();
        let mem = AwwasmMemInst::new(AwwasmMemoryType::new(1, None));
        imports.add_memory(b"env", b"memory", mem);

        let addr = store.store_init(&module, &mut imports).unwrap();
        let inst = store.module(addr).unwrap();
        assert_eq!(inst.memaddrs.len(), 1);
    }

    #[test]
    fn test_instantiate_import_mismatch() {
        let wasm = wat::parse_str(r#"
            (module
                (import "env" "memory" (memory 1))
            )
        "#).unwrap();
        let mut module = AwwasmModule::new(&wasm).unwrap();
        module.resolve_all_sections().unwrap();

        let mut store = AwwasmStore::new();
        let mut imports = AwwasmImports::new();

        let result = store.store_init(&module, &mut imports);
        assert!(result.is_err());
        match result.unwrap_err() {
            AwwasmInstantiationError::MissingImport { module, name } => {
                assert_eq!(module, "env");
                assert_eq!(name, "memory");
            }
            other => panic!("expected MissingImport, got {:?}", other),
        }
    }
}

