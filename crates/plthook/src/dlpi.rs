//! A thin, safe wrapper over `dl_iterate_phdr` — the Linux loader enumeration the
//! ELF backend uses for module acquisition, liveness, and slot-ownership checks
//! unsafe C-callback plumbing exists in exactly one place.

use crate::elf::{self, PHDR_SIZE, ProgramHeaders};
use crate::error::Result;
use core::ffi::{c_int, c_void};
use libc::{dl_iterate_phdr, dl_phdr_info, size_t};

/// One loaded object as reported by `dl_iterate_phdr`: its load bias, its path,
/// and an owned copy of its program-header table bytes (so it safely outlives the
/// callback, which only borrows them).
#[derive(Clone)]
pub(crate) struct Object {
    /// Load bias (`dlpi_addr`).
    pub(crate) base: usize,
    /// Object path (`dlpi_name`); empty for the main executable.
    pub(crate) name: String,
    /// Raw `Elf64_Phdr` table bytes (`phnum * 56`).
    pub(crate) phdrs: Vec<u8>,
    /// Number of program headers.
    pub(crate) phnum: usize,
}

impl Object {
    /// Parse this object's program headers.
    pub(crate) fn program_headers(&self) -> Result<ProgramHeaders> {
        elf::parse_program_headers(&self.phdrs, self.phnum)
    }

    /// Whether `address` falls inside any of this object's loadable segments.
    pub(crate) fn contains_address(&self, address: usize) -> bool {
        let Ok(ph) = self.program_headers() else {
            return false;
        };
        ph.loads.iter().any(|seg| {
            let (Ok(vaddr), Ok(memsz)) = (usize::try_from(seg.vaddr), usize::try_from(seg.memsz))
            else {
                return false;
            };
            let start = self.base.wrapping_add(vaddr);
            let end = start.wrapping_add(memsz);
            address >= start && address < end
        })
    }
}

/// The context threaded through the C callback: the user visitor and the first
/// non-`None` value it yields (which stops the walk early).
struct IterCtx<'a, T> {
    visit: &'a mut dyn FnMut(&Object) -> Option<T>,
    result: Option<T>,
}

/// Visit every loaded object; stop and return the first `Some(_)` the visitor
/// yields, else `None` after visiting all.
pub(crate) fn for_each<T>(mut visit: impl FnMut(&Object) -> Option<T>) -> Option<T> {
    let mut ctx = IterCtx {
        visit: &mut visit,
        result: None,
    };
    // SAFETY: `callback` matches the `dl_iterate_phdr` C ABI; `&mut ctx` is a
    // valid, non-escaping pointer for the duration of the (synchronous) call.
    unsafe {
        dl_iterate_phdr(Some(callback::<T>), (&raw mut ctx).cast::<c_void>());
    }
    ctx.result
}

/// The loaded object with load bias `base`, if any.
pub(crate) fn find_base(base: usize) -> Option<Object> {
    for_each(|obj| (obj.base == base).then(|| obj.clone()))
}

/// The `dl_iterate_phdr` C callback: materialize an [`Object`], hand it to the
/// visitor, and stop the walk (return non-zero) once the visitor yields a value.
extern "C" fn callback<T>(info: *mut dl_phdr_info, _size: size_t, data: *mut c_void) -> c_int {
    if info.is_null() || data.is_null() {
        return 0;
    }
    // SAFETY: `data` is the `&mut IterCtx<T>` handed to `dl_iterate_phdr`.
    let ctx = unsafe { &mut *data.cast::<IterCtx<T>>() };
    // SAFETY: `info` is a valid `dl_phdr_info` provided by the loader.
    let info = unsafe { &*info };
    let phnum = info.dlpi_phnum as usize;
    let bytes = phnum.saturating_mul(PHDR_SIZE);
    let mut phdrs = vec![0u8; bytes];
    if !info.dlpi_phdr.is_null() && bytes != 0 {
        // SAFETY: the loader guarantees `dlpi_phdr` points at `dlpi_phnum`
        // contiguous `Elf64_Phdr` (56 bytes each) for the callback's duration.
        unsafe {
            core::ptr::copy_nonoverlapping(info.dlpi_phdr.cast::<u8>(), phdrs.as_mut_ptr(), bytes);
        }
    }
    let name = if info.dlpi_name.is_null() {
        String::new()
    } else {
        // SAFETY: `dlpi_name` is a NUL-terminated C string owned by the loader.
        unsafe { core::ffi::CStr::from_ptr(info.dlpi_name) }
            .to_string_lossy()
            .into_owned()
    };
    let base = usize::try_from(info.dlpi_addr).unwrap_or(0);
    let obj = Object {
        base,
        name,
        phdrs,
        phnum,
    };
    if let Some(value) = (ctx.visit)(&obj) {
        ctx.result = Some(value);
        return 1;
    }
    0
}
