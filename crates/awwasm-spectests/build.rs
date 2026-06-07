use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let testsuite = PathBuf::from(&manifest_dir).join("../../spec/testsuite");
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir).join("spec_tests.rs");

    // Tell Cargo to rerun this script if the testsuite directory changes.
    println!("cargo:rerun-if-changed={}", testsuite.display());

    if !testsuite.is_dir() {
        fs::write(
            &out_path,
            "// spec/testsuite not initialized.\n\
             // Run: git submodule update --init spec/testsuite\n",
        )
        .unwrap();
        return;
    }

    let mut entries: Vec<_> = fs::read_dir(&testsuite)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |x| x == "wast"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    let mut code = String::new();
    for entry in &entries {
        let path = entry.path();
        let stem = path.file_stem().unwrap().to_string_lossy();
        // Sanitize name: replace non-alphanumeric chars with _ for valid Rust ident.
        let fn_name: String = stem
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let path_str = path.to_string_lossy();
        writeln!(
            code,
            "#[test] fn spec_{fn_name}() {{ run_wast_file(\"{path_str}\"); }}"
        )
        .unwrap();
    }

    fs::write(&out_path, code).unwrap();
    println!("cargo:warning=awwasm-spectests: {} spec test files found", entries.len());
}
