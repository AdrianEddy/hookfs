//! Thin, typed wrappers over the Linux loader / memory syscalls the ELF engine
//! needs. This is the sole Linux-specific surface below the public API; it exposes
//! the same handful of operations the Windows [`crate::sys`](../sys.rs) sibling
//! does (page size, page protection, original-symbol resolution, module liveness,
//! slot ownership) so the platform-agnostic transaction in [`crate::slot`] is
//!
//! Page protection is read from `/proc/self/maps` (there is no `mprotect` "return
//! the old protection" as `VirtualProtect` has), then flipped to writable and
//! restored **exactly** — handling partial/full RELRO GOT pages (R5).

use crate::dlpi;
use crate::error::{Error, Result};
use crate::{ImportKind, Symbol};
use core::ffi::{c_char, c_void};
use std::sync::OnceLock;

/// `errno` immediately after a failed syscall.
fn last_errno() -> u32 {
    std::io::Error::last_os_error()
        .raw_os_error()
        .map_or(0, |e| u32::try_from(e).unwrap_or(0))
}

/// Native page size, queried once (`sysconf(_SC_PAGESIZE)`).
pub(crate) fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        // SAFETY: `sysconf` has no preconditions.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size <= 0 {
            0x1000
        } else {
            usize::try_from(size).unwrap_or(0x1000)
        }
    })
}

/// Current protection flags (`PROT_READ|WRITE|EXEC`) of the region containing
/// `address`, read from `/proc/self/maps`; `0` if the address is unmapped or the
/// map could not be read.
pub(crate) fn query_protection(address: usize) -> u32 {
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return 0;
    };
    for line in maps.lines() {
        let Some((range, rest)) = line.split_once(' ') else {
            continue;
        };
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        let (Ok(start), Ok(end)) = (
            usize::from_str_radix(start, 16),
            usize::from_str_radix(end, 16),
        ) else {
            continue;
        };
        if address >= start && address < end {
            let perms = rest.as_bytes();
            let mut prot = 0u32;
            if perms.first() == Some(&b'r') {
                prot |= libc::PROT_READ as u32;
            }
            if perms.get(1) == Some(&b'w') {
                prot |= libc::PROT_WRITE as u32;
            }
            if perms.get(2) == Some(&b'x') {
                prot |= libc::PROT_EXEC as u32;
            }
            return prot;
        }
    }
    0
}

/// Make a page writable, returning its **original** protection so it can be
/// restored exactly. Reads the current protection from `/proc/self/maps` first
/// (there is no `mprotect` old-protection out-param), then adds `PROT_WRITE` if it
/// is missing (a partial/full RELRO GOT page — R5).
///
/// # Safety
/// `page` must be the start of a mapped page in this process.
pub(crate) unsafe fn make_writable(page: usize) -> Result<u32> {
    let prot = query_protection(page);
    if prot == 0 {
        return Err(Error::Protect {
            slot: page,
            os_error: last_errno(),
        });
    }
    if prot & (libc::PROT_WRITE as u32) == 0 {
        let want = i32::try_from(prot | libc::PROT_WRITE as u32 | libc::PROT_READ as u32)
            .unwrap_or(libc::PROT_READ | libc::PROT_WRITE);
        // SAFETY: `page` is a page-aligned, mapped page; `page_size()` bytes.
        let rc = unsafe { libc::mprotect(page as *mut c_void, page_size(), want) };
        if rc != 0 {
            return Err(Error::Protect {
                slot: page,
                os_error: last_errno(),
            });
        }
    }
    Ok(prot)
}

/// Restore a page's protection to a value previously captured by [`make_writable`].
///
/// # Safety
/// `page` must be the start of a mapped page; `protection` a value from
/// [`make_writable`] for that page.
pub(crate) unsafe fn restore_protection(page: usize, protection: u32) -> Result<()> {
    let prot = i32::try_from(protection).unwrap_or(libc::PROT_READ);
    // SAFETY: `page` is a page-aligned, mapped page; `page_size()` bytes.
    let rc = unsafe { libc::mprotect(page as *mut c_void, page_size(), prot) };
    if rc != 0 {
        return Err(Error::Protect {
            slot: page,
            os_error: last_errno(),
        });
    }
    Ok(())
}

/// Whether an object is still mapped with load bias `base` (the liveness check
pub(crate) fn module_still_loaded(base: usize) -> bool {
    dlpi::find_base(base).is_some()
}

/// Whether `address` lies inside the loadable segments of the object with load
/// bias `module_base` — the extra guard that a slot still belongs to the same
pub(crate) fn slot_belongs_to_module(address: usize, module_base: usize) -> bool {
    dlpi::find_base(module_base).is_some_and(|obj| obj.contains_address(address))
}

/// Resolve the canonical original entry point for an ELF import — the real callee
/// to restore into the slot / call through.
///
/// Always `dlsym(RTLD_DEFAULT, name)`, **never** the slot value: under lazy PLT
/// binding (the shipped BRAW `.so` are partial-RELRO + lazy) a not-yet-called
/// slot holds the resolver stub, so reading it would hand out the trampoline
/// `_slot_value` (the Windows fallback inputs) are unused here.
pub(crate) fn resolve_original(
    _kind: ImportKind,
    _library: &str,
    symbol: &Symbol,
    _slot_value: usize,
) -> Option<usize> {
    let Symbol::Name(name) = symbol else {
        return None;
    };
    let mut c_name: Vec<u8> = name.as_bytes().to_vec();
    c_name.push(0);
    // SAFETY: `c_name` is a valid NUL-terminated C string; `RTLD_DEFAULT` performs
    // the default global lookup.
    let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c_name.as_ptr().cast::<c_char>()) };
    if ptr.is_null() {
        None
    } else {
        Some(ptr as usize)
    }
}
