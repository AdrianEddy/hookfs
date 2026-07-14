//! `plthook` is a pure-Rust import-table rebinding engine.
//!
//! It enumerates imports in loaded Windows PE, Linux ELF, macOS Mach-O, and
//! iOS/iPadOS Mach-O modules. Installing a replacement resolves every target first,
//! changes page protection only as needed, swaps slots atomically, and restores a
//! slot only when the guard still owns it.
//!
//! The binary-format parsers are bounds-checked and host-independent; the live
//! backends provide module discovery, page protection, and original-symbol lookup.

mod arch;
mod error;
mod types;

#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    test
))]
mod import;

#[cfg(any(target_os = "linux", test))]
mod elf;
#[cfg(any(target_os = "macos", target_os = "ios", test))]
mod macho;
#[cfg(any(windows, test))]
mod pe;

pub use error::{Error, Result};
pub use types::{ImportKind, Symbol};

// The live engine: the Windows PE backend, the Linux ELF backend, and the Darwin
// Apple targets). Each supplies a `Module` and a `sys` module exposing the same
// operations the platform-agnostic transaction ([`slot`]) needs; only those
#[cfg(windows)]
mod sys;
#[cfg(target_os = "linux")]
#[path = "sys_unix.rs"]
mod sys;
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[path = "sys_macos.rs"]
mod sys;

#[cfg(target_os = "linux")]
mod dlpi;

#[cfg(windows)]
mod module;
#[cfg(target_os = "linux")]
#[path = "module_elf.rs"]
mod module;
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[path = "module_macho.rs"]
mod module;

#[cfg(any(windows, target_os = "linux", target_os = "macos", target_os = "ios"))]
mod slot;

#[cfg(any(windows, target_os = "linux", target_os = "macos", target_os = "ios"))]
pub use module::Module;
#[cfg(any(windows, target_os = "linux", target_os = "macos", target_os = "ios"))]
pub use slot::{HookGuard, ImportSlot, InstalledHook, Replacement, install};
