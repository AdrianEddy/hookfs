//! Thin, typed wrappers over the Windows loader / memory / process-status
//! syscalls the engine needs. This is the sole Windows-specific surface below
//! the public API; the ELF and Mach-O modules expose the
//! same handful of operations (module lookup, image size, page protection,
//! original-symbol resolution) so the platform-agnostic transaction logic in
//! [`crate::slot`] stays unchanged.

use crate::error::{Error, Result};
use crate::{ImportKind, Symbol};
use core::ffi::c_void;
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::{HANDLE, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    GetModuleFileNameW, GetModuleHandleExW, GetModuleHandleW, GetProcAddress, LoadLibraryW,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_IMAGE, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::ProcessStatus::{
    EnumProcessModules, GetModuleInformation, MODULEINFO,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

/// The current-process pseudo-handle (`(HANDLE)-1`). Constant, needs no syscall,
/// and never has to be closed.
fn current_process() -> HANDLE {
    usize::MAX as HANDLE
}

/// `GetLastError()`; only meaningful immediately after a failed call.
fn last_error() -> u32 {
    // SAFETY: `GetLastError` has no preconditions.
    unsafe { windows_sys::Win32::Foundation::GetLastError() }
}

/// Encode a Rust string as a NUL-terminated UTF-16 buffer for the `*W` APIs.
fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(core::iter::once(0)).collect()
}

/// Native page size, queried once. Used to batch page-protection changes.
pub(crate) fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        // SAFETY: `SYSTEM_INFO` is a plain-old-data struct (integers, pointers,
        // and an integer union) for which an all-zero bit pattern is valid; it is
        // then fully written by `GetSystemInfo`.
        let mut info: SYSTEM_INFO = unsafe { core::mem::zeroed() };
        // SAFETY: `info` is a valid, writable `SYSTEM_INFO`.
        unsafe { GetSystemInfo(&raw mut info) };
        let size = info.dwPageSize as usize;
        if size == 0 { 0x1000 } else { size }
    })
}

/// Resolve a module handle from an address inside it
/// (`GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS`), without changing its refcount.
pub(crate) fn module_from_address(address: *const c_void) -> Result<usize> {
    let mut handle: HMODULE = core::ptr::null_mut();
    // SAFETY: `address` is only used as a lookup key; `handle` is a valid out-ptr.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            address.cast::<u16>(),
            &raw mut handle,
        )
    };
    if ok == 0 || handle.is_null() {
        return Err(Error::ModuleNotFound {
            reason: "address",
            os_error: last_error(),
        });
    }
    Ok(handle as usize)
}

/// Resolve a module handle by name (`GetModuleHandleW`); the module must already
/// be loaded. Does not change its refcount.
pub(crate) fn module_from_name(name: &str) -> Result<usize> {
    let wide = wide(name);
    // SAFETY: `wide` is a valid NUL-terminated UTF-16 string.
    let handle = unsafe { GetModuleHandleW(wide.as_ptr()) };
    if handle.is_null() {
        return Err(Error::ModuleNotFound {
            reason: "name",
            os_error: last_error(),
        });
    }
    Ok(handle as usize)
}

/// `true` if a module is currently mapped at `base` (used for the liveness check
pub(crate) fn module_still_loaded(base: usize) -> bool {
    let mut handle: HMODULE = core::ptr::null_mut();
    // SAFETY: `base` is only a lookup key; `handle` is a valid out-ptr.
    let ok = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            base as *const u16,
            &raw mut handle,
        )
    };
    ok != 0 && handle as usize == base
}

/// Loader-reported mapped size (`SizeOfImage`) of a module — the authoritative
/// bound the parser validates every read against.
pub(crate) fn image_size(base: usize) -> Result<usize> {
    let mut info = MODULEINFO::default();
    // SAFETY: `base` is a candidate module handle; `info` is a valid out struct.
    let ok = unsafe {
        GetModuleInformation(
            current_process(),
            base as HMODULE,
            &raw mut info,
            u32::try_from(core::mem::size_of::<MODULEINFO>()).unwrap_or(u32::MAX),
        )
    };
    if ok == 0 || info.SizeOfImage == 0 {
        return Err(Error::ImageSizeUnknown {
            base,
            os_error: last_error(),
        });
    }
    Ok(info.SizeOfImage as usize)
}

/// Full filesystem path of a module (`GetModuleFileNameW`), lossily decoded.
pub(crate) fn module_path(base: usize) -> String {
    let mut buf = vec![0u16; 260];
    loop {
        let capacity = u32::try_from(buf.len()).unwrap_or(u32::MAX);
        // SAFETY: `buf` is a valid writable buffer of `capacity` u16 elements.
        let len =
            unsafe { GetModuleFileNameW(base as HMODULE, buf.as_mut_ptr(), capacity) as usize };
        if len == 0 {
            return String::new();
        }
        // On truncation the returned length equals the buffer size; grow & retry.
        if len < buf.len() {
            return String::from_utf16_lossy(buf.get(..len).unwrap_or(&buf));
        }
        if buf.len() >= 1 << 16 {
            return String::from_utf16_lossy(&buf);
        }
        buf.resize(buf.len() * 2, 0);
    }
}

/// Enumerate every module currently loaded in this process (`EnumProcessModules`,
pub(crate) fn enumerate_modules() -> Result<Vec<usize>> {
    let process = current_process();
    let mut needed: u32 = 0;
    // First call with a zero-length buffer just to learn the required size.
    // SAFETY: passing a null module buffer with cb == 0 is the documented way to
    // query the needed byte count; `needed` is a valid out-ptr.
    let ok = unsafe { EnumProcessModules(process, core::ptr::null_mut(), 0, &raw mut needed) };
    if ok == 0 {
        return Err(Error::ModuleNotFound {
            reason: "process module list",
            os_error: last_error(),
        });
    }
    let count = needed as usize / core::mem::size_of::<HMODULE>();
    let mut modules: Vec<HMODULE> = vec![core::ptr::null_mut(); count.max(1)];
    let byte_len =
        u32::try_from(modules.len() * core::mem::size_of::<HMODULE>()).unwrap_or(u32::MAX);
    // SAFETY: `modules` has room for `byte_len` bytes; `needed` is a valid out-ptr.
    let ok =
        unsafe { EnumProcessModules(process, modules.as_mut_ptr(), byte_len, &raw mut needed) };
    if ok == 0 {
        return Err(Error::ModuleNotFound {
            reason: "process module list",
            os_error: last_error(),
        });
    }
    let returned = (needed as usize / core::mem::size_of::<HMODULE>()).min(modules.len());
    Ok(modules
        .into_iter()
        .take(returned)
        .map(|m| m as usize)
        .collect())
}

/// `GetProcAddress` for either a named or an ordinal symbol, widened to `usize`.
fn proc_address(module: HMODULE, symbol: &Symbol) -> Option<usize> {
    let proc = match symbol {
        Symbol::Name(name) => {
            let mut c_name: Vec<u8> = name.as_bytes().to_vec();
            c_name.push(0);
            // SAFETY: `c_name` is a valid NUL-terminated byte string; `module` is live.
            unsafe { GetProcAddress(module, c_name.as_ptr()) }
        }
        Symbol::Ordinal(ordinal) => {
            // MAKEINTRESOURCEA: a PCSTR whose high word is zero is an ordinal.
            let by_ordinal = *ordinal as usize as *const u8;
            // SAFETY: `module` is live; the ordinal pointer is never dereferenced
            // by `GetProcAddress`, only interpreted numerically.
            unsafe { GetProcAddress(module, by_ordinal) }
        }
    };
    proc.map(|f| f as usize)
}

/// Resolve the canonical original entry point for a **standard** import — the
/// real callee to restore into the slot / call through — preferring
///
/// Returns `None` if the providing module or symbol can't be resolved; for a
/// standard import the caller then falls back to the pre-write slot value, which
/// is bound to the real callee at load time.
pub(crate) fn resolve_export(library: &str, symbol: &Symbol) -> Option<usize> {
    let wide = wide(library);
    // SAFETY: valid NUL-terminated UTF-16 name.
    let module = unsafe { GetModuleHandleW(wide.as_ptr()) };
    if module.is_null() {
        return None;
    }
    proc_address(module, symbol)
}

/// Authoritatively resolve the original entry point for a **delay-load** import.
///
/// Under `/DELAYLOAD`, a delay IAT slot points at the `__delayLoadHelper2` load
/// thunk until the import is first called; only then does the helper
/// `LoadLibrary` the provider, `GetProcAddress` the symbol, patch the slot, and
/// tail-call. So — unlike a standard import — the pre-write slot value is a
/// resolver stub, never the real callee, and must never be handed out as the
/// original (calling it would overwrite the very slot we patched, silently
/// dropping our hook). We therefore resolve the export directly from the
/// providing DLL: reuse it if already mapped (`GetModuleHandleW`, no refcount
/// change), otherwise force-load it (`LoadLibraryW`) — doing eagerly exactly
/// what the delay-load helper does lazily. Like that helper we deliberately
/// never unload the DLL, so the returned pointer stays valid for as long as any
/// caller may invoke it (the module a live original points into must remain
/// mapped). Returns `None` only if the DLL cannot be loaded or the symbol is
/// absent from it.
pub(crate) fn resolve_delay_export(library: &str, symbol: &Symbol) -> Option<usize> {
    let wide = wide(library);
    // Prefer an already-mapped provider (no refcount change).
    // SAFETY: valid NUL-terminated UTF-16 name.
    let mut module = unsafe { GetModuleHandleW(wide.as_ptr()) };
    if module.is_null() {
        // Not yet loaded: force-load it and keep the reference for the process
        // lifetime (never freed), exactly as `__delayLoadHelper2` would.
        // SAFETY: valid NUL-terminated UTF-16 name.
        module = unsafe { LoadLibraryW(wide.as_ptr()) };
    }
    if module.is_null() {
        return None;
    }
    proc_address(module, symbol)
}

/// Resolve the canonical original entry point for a matched import — the real
/// callee to restore into the slot / call through for passthrough (R6).
///
/// Encapsulates the PE-specific policy so the platform-agnostic transaction in
/// [`crate::slot`] stays uniform (the Linux sibling always resolves via `dlsym`):
/// - **Standard import:** prefer `GetProcAddress`; fall back to `slot_value`, which
///   for a non-delay import is bound to the real callee at load time. Always
///   resolves (never `None`).
/// - **Delay import:** resolve the export authoritatively (force-loading the
///   provider if needed) — the pre-write slot holds only the `__delayLoadHelper2`
///   stub, so there is no sound slot fallback. `None` if the provider/symbol
///   cannot be resolved.
pub(crate) fn resolve_original(
    kind: ImportKind,
    library: &str,
    symbol: &Symbol,
    slot_value: usize,
) -> Option<usize> {
    match kind {
        ImportKind::Standard => Some(resolve_export(library, symbol).unwrap_or(slot_value)),
        ImportKind::DelayLoad => resolve_delay_export(library, symbol),
    }
}

/// Set a page's protection to read/write, returning its previous protection.
///
/// # Safety
/// `page` must be the start of a committed page in this process.
pub(crate) unsafe fn make_writable(page: usize) -> Result<u32> {
    let mut old: u32 = 0;
    // SAFETY: caller guarantees `page` is a committed page; `old` is a valid out-ptr.
    let ok = unsafe {
        VirtualProtect(
            page as *const c_void,
            page_size(),
            PAGE_READWRITE,
            &raw mut old,
        )
    };
    if ok == 0 {
        return Err(Error::Protect {
            slot: page,
            os_error: last_error(),
        });
    }
    Ok(old)
}

/// Restore a page's protection to a previously-captured value.
///
/// # Safety
/// `page` must be the start of a committed page in this process; `protection`
/// must be a value previously returned by [`make_writable`] for that page.
pub(crate) unsafe fn restore_protection(page: usize, protection: u32) -> Result<()> {
    let mut old: u32 = 0;
    // SAFETY: caller guarantees `page` is committed; `old` is a valid out-ptr.
    let ok =
        unsafe { VirtualProtect(page as *const c_void, page_size(), protection, &raw mut old) };
    if ok == 0 {
        return Err(Error::Protect {
            slot: page,
            os_error: last_error(),
        });
    }
    Ok(())
}

/// Current protection flags of the page containing `address`, backing the
/// informational [`crate::ImportSlot::protection`] accessor (queried on demand,
/// never during enumeration). Returns `0` if unavailable.
pub(crate) fn query_protection(address: usize) -> u32 {
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: `address` is only a query key; `info` is a valid out struct.
    let written = unsafe {
        VirtualQuery(
            address as *const c_void,
            &raw mut info,
            core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    if written == 0 { 0 } else { info.Protect }
}

/// `true` if `address` lies in a committed, image-backed region whose allocation
/// base is `module_base` — an extra guard that a slot still belongs to the same
pub(crate) fn slot_belongs_to_module(address: usize, module_base: usize) -> bool {
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: `address` is only a query key; `info` is a valid out struct.
    let written = unsafe {
        VirtualQuery(
            address as *const c_void,
            &raw mut info,
            core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    written != 0
        && info.State == MEM_COMMIT
        && info.Type == MEM_IMAGE
        && info.AllocationBase as usize == module_base
}
