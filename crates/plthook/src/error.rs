//! Typed, thread-safe error model for the import-rebinding engine.
//!
//! The C `plthook` returns integer codes and stashes a description in a
//! out as a defect to fix. Here every fallible operation returns a typed
//! [`Error`] with structured fields — no shared mutable error buffer, no
//! stringly-typed channel — so failures are inspectable and thread-safe.

/// Errors produced by the engine while acquiring, parsing, or patching a module.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A requested module could not be located in the current process
    /// (`GetModuleHandleExW` failed, or a null handle was supplied).
    #[error("module not found: {reason} (os error {os_error})")]
    ModuleNotFound {
        /// What was being resolved (a handle, an address, or a name).
        reason: &'static str,
        /// `GetLastError()` captured at the failure site.
        os_error: u32,
    },

    /// The loader reported no valid mapped size for the module, so no bounded
    /// range exists to parse against.
    #[error("could not determine the mapped size of module at {base:#x} (os error {os_error})")]
    ImageSizeUnknown {
        /// Module base address (`HMODULE`).
        base: usize,
        /// `GetLastError()` captured at the failure site.
        os_error: u32,
    },

    /// A structural defect was found while parsing the mapped image (PE or ELF).
    /// The parser bounds-checks every field against the validated mapped range
    /// *before* dereferencing, so a malformed image yields this error rather than
    #[error("malformed image: {0}")]
    Malformed(&'static str),

    /// The image is a valid PE/ELF the engine intentionally does not handle at
    /// runtime (e.g. a 32-bit PE32 image inside a 64-bit process, a legacy
    /// absolute-VA delay-import descriptor, or an ELF machine that is not
    /// `EM_X86_64`/`EM_AARCH64`). Reported rather than guessed.
    #[error("unsupported image: {0}")]
    Unsupported(&'static str),

    /// A `Replacement` marked `required` matched no import slot in the module.
    #[error("required symbol not found in module `{module}`: {symbol}")]
    SymbolNotFound {
        /// Module name the lookup ran against.
        module: String,
        /// The symbol that was requested but not imported.
        symbol: String,
    },

    /// A matched import's canonical original entry point could not be resolved,
    /// so passing through to it is impossible. This arises for a **delay-load**
    /// import whose providing DLL cannot be loaded (or whose symbol is absent
    /// from it): the delay IAT slot holds only the `__delayLoadHelper2` stub
    /// until first call, and the engine refuses to hand a stub out as the
    /// original (calling it would resolve the import and overwrite the very slot
    /// we patched, silently dropping the hook). Reported for a `required`
    /// replacement; an `optional` one skips the slot instead.
    #[error(
        "could not resolve the original entry point for `{symbol}` from `{library}` \
         (delay-load provider unavailable) in module `{module}`"
    )]
    OriginalUnresolved {
        /// Module whose import was being hooked.
        module: String,
        /// Providing (delay-load) library that could not be resolved.
        library: String,
        /// The symbol whose original could not be resolved.
        symbol: String,
    },

    /// A matched import slot holds an **authenticated** (PAC-signed) pointer — an
    /// it would require a replacement signed with the slot's key, diversity, and
    /// address-discrimination, which cannot be synthesized here; the engine refuses
    /// the slot rather than write a pointer that fails its `AUT*` check (or is
    /// dereferenced unsigned). arm64e support is a dedicated, device-tested effort;
    /// plain arm64 and x86-64 slots are never authenticated and are unaffected.
    #[error(
        "refusing to rebind authenticated (arm64e PAC-signed) import `{symbol}` in \
         module `{module}`: authenticated slots require dedicated PAC support"
    )]
    AuthenticatedSlot {
        /// Module whose authenticated import was matched.
        module: String,
        /// The symbol whose slot is authenticated.
        symbol: String,
    },

    /// Changing a slot page's protection failed. Carries the slot address and
    /// the OS error. When this happens mid-install the engine rolls back every
    #[error("VirtualProtect failed for slot {slot:#x}: os error {os_error}")]
    Protect {
        /// Address of the import slot whose page could not be reprotected.
        slot: usize,
        /// `GetLastError()` from the failing `VirtualProtect` call.
        os_error: u32,
    },

    /// During restore, a slot no longer held this guard's replacement pointer,
    /// untouched and reports the conflict instead (compare-exchange restore,
    #[error(
        "restore conflict at slot {slot:#x}: expected {expected:#x}, found {found:#x} \
         (a subsequent hook owns this slot; left untouched)"
    )]
    RestoreConflict {
        /// Address of the conflicted import slot.
        slot: usize,
        /// The replacement pointer this guard installed and expected to find.
        expected: usize,
        /// The pointer actually present in the slot.
        found: usize,
    },
}

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;
