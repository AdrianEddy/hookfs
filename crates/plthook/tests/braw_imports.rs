//! Enumerate a real Blackmagic RAW SDK DLL with the runtime engine and assert its
//! (`audit/manifests/windows/BlackmagicRawAPI.dll.import.json`).
//!
//! The DLL is mapped with `DONT_RESOLVE_DLL_REFERENCES` — a real image mapping
//! (so RVA == memory offset, as the engine assumes) but without running the SDK's
//! `DllMain` or resolving its dependencies. The Import Name Table is present
//! regardless, so names/ordinals enumerate exactly as they do from disk.
//!
//! If the SDK DLL or the manifest is absent (e.g. CI without the SDK checkout),
//! the test is skipped rather than failed.
#![cfg(windows)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use plthook::{Module, Symbol};
use std::collections::BTreeSet;
use windows_sys::Win32::Foundation::HMODULE;
use windows_sys::Win32::System::LibraryLoader::{DONT_RESOLVE_DLL_REFERENCES, LoadLibraryExW};

const DLL_PATH: &str =
    r"d:\programowanie\projekty\Rust\braw-rs\sdk\Win\Libraries\BlackmagicRawAPI.dll";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// `(library, symbol-description)` — the comparable identity of one import,
/// matching the manifest's `library` + `name` (`#N` for ordinals).
fn key(library: &str, symbol: &str) -> String {
    format!("{library}|{symbol}")
}

#[test]
fn engine_import_set_matches_manifest() {
    let manifest_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        r"\..\..\audit\manifests\windows\BlackmagicRawAPI.dll.import.json"
    );
    let Ok(manifest_text) = std::fs::read_to_string(manifest_path) else {
        eprintln!("skipping: manifest not found at {manifest_path}");
        return;
    };

    let wide = wide(DLL_PATH);
    // SAFETY: `wide` is a valid NUL-terminated path; the flag maps the image
    // without running DllMain or resolving imports.
    let handle: HMODULE = unsafe {
        LoadLibraryExW(
            wide.as_ptr(),
            std::ptr::null_mut(),
            DONT_RESOLVE_DLL_REFERENCES,
        )
    };
    if handle.is_null() {
        eprintln!("skipping: could not load {DLL_PATH}");
        return;
    }

    // SAFETY: `handle` is a live image mapping from `LoadLibraryExW`.
    let module = unsafe { Module::from_handle(handle.cast()) }.expect("acquire BlackmagicRawAPI");
    let imports = module.imports().expect("enumerate imports");

    // Actual set from the live engine.
    let actual: BTreeSet<String> = imports
        .iter()
        .map(|slot| {
            let symbol = slot
                .symbol()
                .map_or_else(|| "<address-only>".to_owned(), Symbol::describe);
            key(slot.library(), &symbol)
        })
        .collect();

    let manifest: serde_json::Value = serde_json::from_str(&manifest_text).unwrap();
    let image = &manifest["images"][0];
    let expected: BTreeSet<String> = image["imports"]
        .as_array()
        .expect("imports array")
        .iter()
        .map(|import| {
            let library = import["library"].as_str().unwrap_or_default();
            let name = import["name"].as_str().unwrap_or_default();
            key(library, name)
        })
        .collect();

    let missing: Vec<_> = expected.difference(&actual).collect();
    let extra: Vec<_> = actual.difference(&expected).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "engine import set differs from the manifest\n  missing (in manifest, not enumerated): {missing:?}\n  extra (enumerated, not in manifest): {extra:?}",
    );

    for symbol in [
        "CreateFileW",
        "ReadFile",
        "SetFilePointerEx",
        "GetFileSizeEx",
        "CloseHandle",
    ] {
        assert!(
            imports
                .iter()
                .any(|s| s.symbol() == Some(&Symbol::name(symbol))
                    && s.library().eq_ignore_ascii_case("KERNEL32.dll")),
            "expected KERNEL32!{symbol} among the enumerated imports",
        );
    }

    // The manifest records this DLL as importing no CRT and having no delay
    // imports; confirm the engine sees the same shape.
    assert_eq!(expected.len(), 161, "manifest import count");
    assert_eq!(actual.len(), 161, "engine import count");
}
