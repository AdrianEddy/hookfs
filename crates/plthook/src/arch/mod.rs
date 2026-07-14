//! Per-architecture and per-format constants.
//!
//! The only genuinely machine-dependent facts the engine needs are the format's
//! machine identifier (COFF `Machine` for PE, `e_machine` for ELF) and the native
//! pointer width (import slots are machine-pointer-sized). PE and ELF constants are
//! cfg-gated to the build that actually uses them (plus `test`, so the
//! host-independent parser tests compile on either host) so neither warns as dead
//! code on the other platform's non-test build.

/// Width in bytes of a native code pointer (and of a PE32+ IAT thunk, an ELF
/// GOT slot, or a Mach-O symbol-pointer / fixup slot). Used by the
/// platform-agnostic transaction to page-batch the writes. Gated to the backends
/// that own a live engine (`slot`/`module`); other targets build only the format
/// parsers, which never consult it.
#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "macos",
    target_os = "ios",
    test
))]
pub(crate) const PTR_SIZE: usize = core::mem::size_of::<usize>();

// ---- PE / COFF -------------------------------------------------------------

/// COFF `IMAGE_FILE_HEADER.Machine` value for x64.
#[cfg(any(windows, test))]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
/// COFF `IMAGE_FILE_HEADER.Machine` value for ARM64.
#[cfg(any(windows, test))]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) const IMAGE_FILE_MACHINE_ARM64: u16 = 0xAA64;

/// The COFF machine identifier of the process this engine is compiled into.
///
/// A module loaded into a running process always matches the process machine,
/// so the parser validates a freshly-acquired image against this value as a
/// cheap sanity check on the parsed header (32-bit PE32 images are rejected
/// separately, by optional-header magic).
#[cfg(all(any(windows, test), target_arch = "x86_64"))]
pub(crate) const HOST_MACHINE: u16 = IMAGE_FILE_MACHINE_AMD64;
#[cfg(all(any(windows, test), target_arch = "aarch64"))]
pub(crate) const HOST_MACHINE: u16 = IMAGE_FILE_MACHINE_ARM64;

// ---- ELF -------------------------------------------------------------------

/// ELF `e_machine` for x86-64.
#[cfg(any(target_os = "linux", test))]
pub(crate) const EM_X86_64: u16 = 62;
/// ELF `e_machine` for `AArch64`.
#[cfg(any(target_os = "linux", test))]
pub(crate) const EM_AARCH64: u16 = 183;

/// The relocation types that mark an import slot for one ELF machine: the PLT
/// jump-slot (`.rela.plt`) and the GOT data slot (`.rela.dyn`). Both are pointer
/// cells the engine may rebind.
#[cfg(any(target_os = "linux", test))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ElfRelocKinds {
    /// `R_<arch>_JUMP_SLOT` — a `.rela.plt` PLT/GOT slot (function imports).
    pub(crate) jump_slot: u32,
    /// `R_<arch>_GLOB_DAT` — a `.rela.dyn` GOT data slot (data/function-address
    /// imports).
    pub(crate) glob_dat: u32,
}

/// The jump-slot / glob-dat relocation types for an ELF `e_machine`, or `None`
/// for a machine this engine does not support. Keyed off the *image's* machine
/// (not the host arch) so the offline parser can decode an x86-64 or `AArch64`
/// object on any host.
#[cfg(any(target_os = "linux", test))]
pub(crate) const fn elf_reloc_kinds(e_machine: u16) -> Option<ElfRelocKinds> {
    match e_machine {
        // R_X86_64_JUMP_SLOT = 7, R_X86_64_GLOB_DAT = 6.
        EM_X86_64 => Some(ElfRelocKinds {
            jump_slot: 7,
            glob_dat: 6,
        }),
        // R_AARCH64_JUMP_SLOT = 1026, R_AARCH64_GLOB_DAT = 1025.
        EM_AARCH64 => Some(ElfRelocKinds {
            jump_slot: 1026,
            glob_dat: 1025,
        }),
        _ => None,
    }
}

// ---- Mach-O -----------------------------------------------------------------
//
// use the identical Mach-O engine; the constants are gated to `any(target_os =
// "macos", target_os = "ios", test)` so they build on either Apple target and on any
// host under `test` (the parser is host-independent).

/// `cpu_type_t` mask marking the 64-bit ABI.
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) const CPU_ARCH_ABI64: u32 = 0x0100_0000;
/// Mach-O `cputype` for x86-64 (`CPU_TYPE_X86 | CPU_ARCH_ABI64`).
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) const CPU_TYPE_X86_64: u32 = 0x7 | CPU_ARCH_ABI64;
/// Mach-O `cputype` for arm64 / arm64e (`CPU_TYPE_ARM | CPU_ARCH_ABI64`).
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) const CPU_TYPE_ARM64: u32 = 0xc | CPU_ARCH_ABI64;
/// The `cpusubtype` mask that isolates the subtype from the capability bits
/// (`CPU_SUBTYPE_MASK` covers the high 8 bits, e.g. the "lib64" capability).
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) const CPU_SUBTYPE_MASK: u32 = 0xff00_0000;
/// arm64e `cpusubtype` (`CPU_SUBTYPE_ARM64E`) — authenticated-pointer arm64. The
/// standard iOS/iPadOS device target is plain arm64; arm64e (PAC-signed) slots are
/// The parser still reports every other ordinary slot.
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) const CPU_SUBTYPE_ARM64E: u32 = 2;

/// Whether the engine supports rebinding a Mach-O image of this `cputype`. Only
/// the two 64-bit userland architectures the SDK ships (x86-64 + arm64) are
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) const fn macho_cpu_supported(cputype: u32) -> bool {
    matches!(cputype, CPU_TYPE_X86_64 | CPU_TYPE_ARM64)
}
