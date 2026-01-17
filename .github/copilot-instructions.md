# AwWasm Copilot Instructions

## Project Overview

AwWasm is a minimal WebAssembly runtime written in **safe Rust** (no `unsafe` code), designed to be `no_std` compatible and portable. This is a learning-focused project emphasizing safety and simplicity.

## Architecture

### Workspace Structure
- **Root workspace**: Cargo workspace coordinating two crates via `members = ["crates/*"]`
- **`awwasm-parser`**: WASM binary parser built with `nom` parser combinators (git submodule from `deepkn/awwasm-parser`)
- **`awwasm-runtime`**: Runtime execution engine (in early development)

### Parser Design (`crates/awwasm-parser`)

The parser uses a **two-phase parsing strategy**:

1. **Initial parse**: `AwwasmModule::new()` parses section headers only, storing raw `section_body: &[u8]` for each section
2. **Resolution**: `resolve_all_sections()` parses section bodies on-demand, populating typed fields like `types`, `funcs`, `code`, etc.

**Key modules** (in `src/components/`):
- `module.rs`: Top-level `AwwasmModule` with preamble (magic number + version) and sections
- `section.rs`: Section enumeration (`Type`, `Import`, `Function`, `Code`, `Memory`, `Export`, `Data`), header parsing, and resolution via `SectionItem` enum
- `types.rs`: All WASM type definitions - function signatures, imports, exports, memory, code items
- `instructions.rs`: Instruction opcodes (minimal stub currently)

### Parser Component Relationships

```
AwwasmModule
  ├── AwwasmModulePreamble (magic: b"\0asm", version: u32)
  └── Vec<AwwasmSection>
        ├── AwwasmSectionHeader (section_type, section_size)
        └── section_body: &[u8] → resolved via resolve() into:
              ├── Vec<AwwasmTypeSectionItem> (function signatures)
              ├── Vec<AwwasmFuncSectionItem> (type indices)
              ├── Vec<AwwasmCodeSectionItem> → resolve() → AwwasmFunction
              ├── Vec<AwwasmImportSectionItem>
              ├── Vec<AwwasmExportSectionItem>
              ├── Vec<AwwasmMemorySectionItem>
              └── Vec<AwwasmDataSectionItem>
```

## Critical Conventions

### 1. `no_std` Compatibility Requirement
ALL dependencies in `awwasm-parser` MUST specify `default-features=false` to maintain `no_std` compatibility:
```toml
nom = {version="7.1.3", default-features=false}
anyhow = {version="1.0.71", default-features=false}
```
Never add std-dependent features without this configuration.

### 2. Parsing with `nom` and `nom-derive`

Use the `#[derive(Nom)]` macro pattern for declarative parsing:
```rust
#[derive(Debug, Clone, PartialEq, Eq, Nom)]
#[nom(LittleEndian)]
pub struct AwwasmTypeSectionItem<'a> {
    #[nom(Tag(WASM_TYPE_SECTION_OPCODE_FUNC))]  // Validates magic bytes
    pub type_magic: &'a[u8],
    #[nom(LengthCount="leb128_u32")]            // Parse u32 count, then that many items
    pub fn_args: Vec<ParamType>,
    #[nom(Parse="leb128_u32")]                   // Custom parser function
    pub section_size: u32,
    #[nom(Take="section_size")]                  // Take N bytes
    pub section_body: &'a[u8],
}
```

**LEB128 encoding**: WASM uses variable-length unsigned integers (`leb128_u32` from `nom-leb128`). Sizes in section headers are LEB128 encoded and must account for the encoding overhead when slicing bytes.

### 3. Zero-Copy Lifetime Pattern
Parsed structures borrow from input with lifetime `'a`:
```rust
pub struct AwwasmModule<'a> { /* borrows from original &[u8] */ }
```
This avoids allocations. Never clone byte slices unnecessarily.

### 4. Testing Workflow

Tests use `wat::parse_str()` to convert WebAssembly Text (WAT) to binary for parsing:
```rust
#[test]
fn decode_function_signature_test() -> Result<()> {
    let module = wat::parse_str("(module (func (param i32 i64)))")?;
    let mut module_parsed = AwwasmModule::new(&module)?;
    module_parsed.resolve_all_sections()?;
    assert_eq!(module_parsed.types, Some(vec![...]));
    Ok(())
}
```

**Run tests**: `cargo test` (workspace-wide) or `cargo test -p awwasm-parser` (single crate)

## Developer Workflows

### Build & Test
```bash
cargo build              # Build all workspace members
cargo test               # Run all tests
cargo test -p awwasm-parser  # Test parser crate only
cargo build --release    # Release build
```

### Working with Submodules
The parser is a git submodule. When pulling changes:
```bash
git submodule update --init --recursive
```

To update the submodule to latest:
```bash
cd crates/awwasm-parser
git pull origin main
cd ../..
git add crates/awwasm-parser
git commit -m "Bump to latest version of awwasm-parser"
```

### Debugging WASM Binaries
Inspect binary bytes vs. test expectations:
```rust
println!("Raw bytes: {:?}", &module[..16]);  // Print first 16 bytes
```

## Key Files Reference

- `crates/awwasm-parser/src/limits.rs`: Mozilla-derived WASM validation limits (MAX_WASM_FUNCTIONS, MAX_WASM_MEMORY32_PAGES, etc.)
- `crates/awwasm-parser/src/consts.rs`: WASM magic constants (magic number `b"\0asm"`, opcode bytes)
- `crates/awwasm-parser/Cargo.toml`: All dependencies with `default-features=false` - reference this pattern for new deps

## When Adding Features

1. **New WASM sections**: Add enum variant to `SectionCode`, create types in `types.rs`, add resolution branch in `section.rs:resolve()`
2. **New instructions**: Extend `Instruction` enum in `instructions.rs` with numeric opcodes matching WASM spec
3. **New tests**: Use `wat::parse_str()` + `pretty_assertions` for clear diffs
4. **Error handling**: Use `anyhow::Result` and `map_err()` with context strings like `"Failed to parse WASM Type Section: {}"`

## Safety & Constraints

- **NO `unsafe` blocks allowed** - this is a hard project rule
- **`no_std` forever** - all code must work in embedded/bare-metal environments
- Favor simplicity over optimization - this is a learning project, not production runtime
