//! Typed Darwin loader and Mach VM wrappers used by the Mach-O backend.
//!
//! macOS and iOS/iPadOS share these operations: module discovery, symbol lookup,
//! page-protection changes, image-liveness checks, and slot ownership checks.
//! Const data slots are temporarily made writable with copy-on-write protection;
//! executable pages and authenticated arm64e slots are never written.

use crate::error::{Error, Result};
use crate::{ImportKind, Symbol};
use core::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

use mach2::dyld;
use mach2::kern_return::KERN_SUCCESS;
use mach2::traps::mach_task_self;
use mach2::vm::{mach_vm_protect, mach_vm_region};
use mach2::vm_prot::{VM_PROT_COPY, VM_PROT_EXECUTE, VM_PROT_READ, VM_PROT_WRITE};
use mach2::vm_region::{VM_REGION_BASIC_INFO_64, vm_region_basic_info_data_64_t};

/// Reinterpret a `kern_return_t` as the [`Error::Protect`] `os_error` field (mach
/// calls do not set `errno`; the raw kern code is the useful diagnostic).
fn kr_code(kr: c_int) -> u32 {
    u32::from_ne_bytes(kr.to_ne_bytes())
}

/// The R/W/X protection bits, as a `u32` (`VM_PROT_*` bit values coincide with the
/// POSIX `PROT_*` ones), used uniformly by the transaction as an opaque round-trip
/// token between [`make_writable`] and [`restore_protection`].
fn rwx_mask() -> u32 {
    u32::from_ne_bytes((VM_PROT_READ | VM_PROT_WRITE | VM_PROT_EXECUTE).to_ne_bytes())
}

/// Convert a captured protection `u32` back to a `vm_prot_t` for a mach call. The
/// value only ever carries the low R/W/X bits, so the conversion is lossless.
fn as_vm_prot(protection: u32) -> c_int {
    c_int::try_from(protection & rwx_mask()).unwrap_or(VM_PROT_READ)
}

/// Native page size, queried once (`sysconf(_SC_PAGESIZE)`).
pub(crate) fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    *PAGE_SIZE.get_or_init(|| {
        // SAFETY: `sysconf` has no preconditions.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if size <= 0 {
            0x4000
        } else {
            usize::try_from(size).unwrap_or(0x4000)
        }
    })
}

/// Current protection flags (`PROT_READ|WRITE|EXEC`) of the region containing
/// `address`, from `mach_vm_region`; `0` if the address is unmapped or the query
/// fails. `mach_vm_region` returns the region at *or above* `address`, so the result
/// is accepted only when it actually contains `address`.
pub(crate) fn query_protection(address: usize) -> u32 {
    // SAFETY: reads the (static) task-self port; no preconditions.
    let task = unsafe { mach_task_self() };
    let mut region_addr = address as u64;
    let mut region_size: u64 = 0;
    // SAFETY: `vm_region_basic_info_64` is plain-old-data; all-zero is valid.
    let mut info: vm_region_basic_info_data_64_t = unsafe { core::mem::zeroed() };
    let mut count = vm_region_basic_info_data_64_t::count();
    let mut object_name: mach2::port::mach_port_t = 0;
    // SAFETY: all out-pointers are valid; `info`/`count` describe the flavor buffer.
    let kr = unsafe {
        mach_vm_region(
            task,
            &raw mut region_addr,
            &raw mut region_size,
            VM_REGION_BASIC_INFO_64,
            (&raw mut info).cast::<c_int>(),
            &raw mut count,
            &raw mut object_name,
        )
    };
    if kr != KERN_SUCCESS {
        return 0;
    }
    let addr = address as u64;
    if addr < region_addr || addr >= region_addr.saturating_add(region_size) {
        return 0; // `address` fell in a hole before the next mapped region
    }
    u32::from_ne_bytes(info.protection.to_ne_bytes()) & rwx_mask()
}

/// Make a page writable, returning its **original** protection so it can be restored
/// exactly. Queries the current protection first (there is no old-protection
/// out-param), then adds write A?€�t using `VM_PROT_COPY` so a read-only const page
/// (`__DATA_CONST`/`__AUTH_CONST`) can be raised past its `max_protection`.
///
/// # Safety
/// `page` must be the start of a mapped page in this process.
pub(crate) unsafe fn make_writable(page: usize) -> Result<u32> {
    let prot = query_protection(page);
    if prot == 0 {
        return Err(Error::Protect {
            slot: page,
            os_error: 0,
        });
    }
    if prot & u32::from_ne_bytes(VM_PROT_WRITE.to_ne_bytes()) == 0 {
        // SAFETY: task-self port; no preconditions.
        let task = unsafe { mach_task_self() };
        let base = as_vm_prot(prot) | VM_PROT_READ | VM_PROT_WRITE;
        // Prefer a copy-on-write raise (lifts a const page past max_protection).
        // SAFETY: `page`/`page_size()` describe a mapped page range we own.
        let kr = unsafe {
            mach_vm_protect(
                task,
                page as u64,
                page_size() as u64,
                0,
                base | VM_PROT_COPY,
            )
        };
        if kr != KERN_SUCCESS {
            // Some regions reject VM_PROT_COPY; fall back to a plain raise.
            // SAFETY: same page range.
            let kr2 = unsafe { mach_vm_protect(task, page as u64, page_size() as u64, 0, base) };
            if kr2 != KERN_SUCCESS {
                return Err(Error::Protect {
                    slot: page,
                    os_error: kr_code(kr2),
                });
            }
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
    // SAFETY: task-self port; no preconditions.
    let task = unsafe { mach_task_self() };
    // SAFETY: `page`/`page_size()` describe a mapped page range we own.
    let kr = unsafe {
        mach_vm_protect(
            task,
            page as u64,
            page_size() as u64,
            0,
            as_vm_prot(protection),
        )
    };
    if kr != KERN_SUCCESS {
        return Err(Error::Protect {
            slot: page,
            os_error: kr_code(kr),
        });
    }
    Ok(())
}

/// Whether an image is still mapped with its Mach-O header at `base` (the liveness
/// list for a header address equal to `base`.
pub(crate) fn module_still_loaded(base: usize) -> bool {
    // SAFETY: `_dyld_image_count` / `_dyld_get_image_header` have no preconditions.
    let count = unsafe { dyld::_dyld_image_count() };
    (0..count).any(|i| unsafe { dyld::_dyld_get_image_header(i) } as usize == base)
}

/// Whether `address` lies inside the mapped vm span of the image whose header is at
/// `module_base` A?€�t the extra guard that a slot still belongs to the same image
/// unload.
pub(crate) fn slot_belongs_to_module(address: usize, module_base: usize) -> bool {
    // SAFETY: dyld image accessors have no preconditions.
    let count = unsafe { dyld::_dyld_image_count() };
    for i in 0..count {
        // SAFETY: `i < count`.
        if unsafe { dyld::_dyld_get_image_header(i) } as usize != module_base {
            continue;
        }
        // SAFETY: `i < count`.
        let slide = unsafe { dyld::_dyld_get_image_vmaddr_slide(i) };
        return image_contains(module_base, slide, address);
    }
    false
}

/// Walk the in-memory `mach_header_64` at `header` (a live, dyld-mapped image) and
/// report whether `slide + vmaddr .. + vmsize` of any `LC_SEGMENT_64` covers
/// `address`. Reads are bounded to the validated command region, so a corrupt (but
/// mapped) header cannot walk off the mapping.
fn image_contains(header: usize, slide: isize, address: usize) -> bool {
    /// `LC_SEGMENT_64`.
    const LC_SEGMENT_64: u32 = 0x19;
    /// `MH_MAGIC_64`.
    const MH_MAGIC_64: u32 = 0xfeed_facf;
    /// Bound on `ncmds` / `sizeofcmds` to keep a corrupt header in-bounds.
    const MAX_CMDS: u32 = 1 << 16;

    // SAFETY: `header` is a live, mapped `mach_header_64` supplied by dyld; the first
    // 32 bytes are always readable. `read_unaligned` avoids any alignment assumption.
    let magic = unsafe { core::ptr::read_unaligned(header as *const u32) };
    if magic != MH_MAGIC_64 {
        return false;
    }
    // ncmds@16, sizeofcmds@20.
    // SAFETY: within the 32-byte header.
    let ncmds = unsafe { core::ptr::read_unaligned((header + 16) as *const u32) };
    // SAFETY: within the 32-byte header.
    let sizeofcmds = unsafe { core::ptr::read_unaligned((header + 20) as *const u32) } as usize;
    if ncmds > MAX_CMDS {
        return false;
    }
    let cmds_end = header.saturating_add(32).saturating_add(sizeofcmds);
    let mut cursor = header + 32;
    for _ in 0..ncmds {
        if cursor.saturating_add(8) > cmds_end {
            break;
        }
        // SAFETY: `[cursor, cursor+8)` is within the command region.
        let cmd = unsafe { core::ptr::read_unaligned(cursor as *const u32) };
        // SAFETY: as above.
        let cmdsize = unsafe { core::ptr::read_unaligned((cursor + 4) as *const u32) } as usize;
        if cmdsize < 8 || cursor.saturating_add(cmdsize) > cmds_end {
            break;
        }
        if cmd == LC_SEGMENT_64 && cmdsize >= 72 {
            // segment_command_64: vmaddr@24, vmsize@32.
            // SAFETY: the command spans `cmdsize >= 72` bytes inside the region.
            let vmaddr_u64 = unsafe { core::ptr::read_unaligned((cursor + 24) as *const u64) };
            // SAFETY: as above.
            let vmsize_u64 = unsafe { core::ptr::read_unaligned((cursor + 32) as *const u64) };
            let vmaddr = usize::try_from(vmaddr_u64).unwrap_or(usize::MAX);
            let vmsize = usize::try_from(vmsize_u64).unwrap_or(0);
            let start = vmaddr.wrapping_add_signed(slide);
            let end = start.wrapping_add(vmsize);
            if address >= start && address < end {
                return true;
            }
        }
        cursor = cursor.saturating_add(cmdsize);
    }
    false
}

/// Resolve the canonical original entry point for a matched Mach-O import A?€�t the real
/// callee to restore into the slot / call through for passthrough (R6).
///
/// 1. **Prefer the pre-write slot value** when it genuinely resolves to the target
///    function (validated via `dladdr`, name-normalized). For an eagerly-bound slot
///    A?€�t a chained-fixups image or a non-lazy `__got` entry, the common modern case
///    and what the BRAW framework uses A?€�t this is the exact pointer the image is
///    bound to, including the precise x86-64 `$INODE64` stat variant, so passthrough
///    is byte-for-byte correct.
/// 2. Otherwise `dlsym(RTLD_DEFAULT, name)` (the base name; `dlsym` re-adds the
///    leading `_`) A?€�t always a real export, never a lazy stub. This is the canonical
///    aliasing).
/// 3. As a last resort, the raw slot value.
///
/// `_kind` / `_library` are the PE/ELF inputs, unused here (Mach-O matches by name
/// across any provider).
pub(crate) fn resolve_original(
    _kind: ImportKind,
    _library: &str,
    symbol: &Symbol,
    slot_value: usize,
) -> Option<usize> {
    let Symbol::Name(name) = symbol else {
        return None;
    };

    // 1. The exact bound pointer, when it really is `name` (not a lazy stub).
    if slot_value != 0 && dladdr_name_matches(slot_value, name) {
        return Some(slot_value);
    }

    // 2. The canonical current export.
    let mut c_name: Vec<u8> = name.as_bytes().to_vec();
    c_name.push(0);
    // SAFETY: `c_name` is a valid NUL-terminated C string; `RTLD_DEFAULT` performs
    // the default global lookup (adding the Mach-O leading underscore itself).
    let ptr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, c_name.as_ptr().cast::<c_char>()) };
    if !ptr.is_null() {
        return Some(ptr as usize);
    }

    // 3. Last resort: whatever the slot held (better than dropping a required hook).
    (slot_value != 0).then_some(slot_value)
}

/// Whether `addr` resolves (via `dladdr`) to a symbol whose normalized name equals
/// `name` A?€�t i.e. the slot really points at the import, not a binding stub.
fn dladdr_name_matches(addr: usize, name: &str) -> bool {
    // SAFETY: `Dl_info` is plain-old-data; all-zero is valid until `dladdr` fills it.
    let mut info: libc::Dl_info = unsafe { core::mem::zeroed() };
    // SAFETY: `addr` is only used as a lookup key; `info` is a valid out-pointer.
    let ok = unsafe { libc::dladdr(addr as *const c_void, &raw mut info) };
    if ok == 0 || info.dli_sname.is_null() {
        return false;
    }
    // SAFETY: `dli_sname` is a NUL-terminated C string owned by the loader.
    let sname = unsafe { core::ffi::CStr::from_ptr(info.dli_sname) };
    sname
        .to_str()
        .ok()
        .and_then(crate::macho::normalize_symbol_name)
        .is_some_and(|resolved| resolved == name)
}
