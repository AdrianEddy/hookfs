//! [`Module`](crate::Module).
//!
//! A [`Module`] is a validated identity for one object currently mapped into the
//! process: its load bias (`dl_iterate_phdr`'s `dlpi_addr`), its loadable
//! segments, the address of its dynamic table, and its path. Acquisition walks the
//! program headers and validates the ELF header up front, so an address that names
//! no loaded object — or a non-ELF64 one — is rejected here rather than deep inside
//! enumeration.
//!
//! The heavy lifting (relocation / symbol / version decoding) lives in the
//! host-independent [`crate::elf`] parser; this module only supplies the *live*
//! [`ElfView`](crate::elf::ElfView): dynamic pointers are absolute addresses (the
//! glibc dynamic linker relocates the dynamic section in place on x86-64/`AArch64`),
//! and a slot address is `load_bias + r_offset`.

use crate::dlpi::{self, Object};
use crate::elf::{self, ElfView, Segment};
use crate::error::{Error, Result};
use crate::slot::ImportSlot;
use core::ffi::{c_char, c_void};

/// The head of glibc's `struct link_map`. Only the first field — the load bias
/// `l_addr` (`ElfW(Addr)`) — is read; `dlinfo(RTLD_DI_LINKMAP)` writes a pointer to
/// the real, longer `link_map`, whose tail is intentionally opaque here. The
/// `libc` crate does not expose `link_map`, so the minimal ABI head is declared.
#[repr(C)]
struct LinkMapHead {
    l_addr: usize,
}

/// A validated, currently-loaded ELF object.
#[derive(Debug, Clone)]
pub struct Module {
    /// Load bias (`dlpi_addr`) — added to a link-time vaddr to get its address.
    base: usize,
    /// Loadable segments, for bounds-checking live reads and slot targets.
    loads: Vec<Segment>,
    /// The image's `e_machine` (picks the relocation constants).
    machine: u16,
    /// Absolute address of the dynamic (`PT_DYNAMIC`) table.
    dynamic_addr: u64,
    /// Lowest / highest mapped byte (`[lo, hi)`), for size + liveness checks.
    span: (usize, usize),
    /// The object's path (`dlpi_name`), possibly empty for the main executable.
    path: String,
}

/// Read `buf.len()` bytes from live process memory at absolute address `cursor`,
/// bounds-checked to lie inside a mapped `PT_LOAD` segment of the object with load
/// bias `base`.
fn read_live(base: usize, loads: &[Segment], cursor: u64, buf: &mut [u8]) -> Result<()> {
    let len = buf.len() as u64;
    // `cursor` is an absolute address; recover the link-time vaddr to bound-check.
    let vaddr = cursor
        .checked_sub(base as u64)
        .ok_or(Error::Malformed("address below load base"))?;
    if !vaddr_in_load(loads, vaddr, len) {
        return Err(Error::Malformed("read outside loadable segments"));
    }
    let addr = usize::try_from(cursor).map_err(|_| Error::Malformed("address overflow"))?;
    // SAFETY: `[cursor, cursor+len)` was validated to lie within a committed,
    // readable `PT_LOAD` mapping of this live object; `buf` cannot overlap it.
    unsafe {
        core::ptr::copy_nonoverlapping(addr as *const u8, buf.as_mut_ptr(), buf.len());
    }
    Ok(())
}

/// Whether `[vaddr, vaddr+len)` lies inside some loadable segment's in-memory
/// range (`p_memsz` covers `.bss`, which is mapped and readable).
fn vaddr_in_load(loads: &[Segment], vaddr: u64, len: u64) -> bool {
    loads.iter().any(|seg| {
        let end = seg.vaddr.saturating_add(seg.memsz);
        vaddr >= seg.vaddr && vaddr.checked_add(len).is_some_and(|hi| hi <= end)
    })
}

impl ElfView for Module {
    fn machine(&self) -> u16 {
        self.machine
    }
    fn dynamic_cursor(&self) -> u64 {
        self.dynamic_addr
    }
    fn dynptr_cursor(&self, d_ptr: u64) -> Result<u64> {
        // The dynamic linker has relocated the *address* dynamic pointers
        // (DT_SYMTAB/DT_STRTAB/DT_JMPREL/DT_RELA/DT_VERSYM) to absolute addresses
        // in place, so `d_ptr` is already the address to read (no bias added). It
        // must fall within a loadable range, or the image is malformed.
        let vaddr = d_ptr
            .checked_sub(self.base as u64)
            .ok_or(Error::Malformed("dynamic pointer below load base"))?;
        if !vaddr_in_load(&self.loads, vaddr, 1) {
            return Err(Error::Malformed(
                "dynamic pointer outside loadable segments",
            ));
        }
        Ok(d_ptr)
    }
    fn vaddr_cursor(&self, vaddr: u64) -> Result<u64> {
        // DT_VERNEED / DT_VERDEF are NOT relocated in place by glibc's
        // ADJUST_DYN_INFO (unlike the pointers `dynptr_cursor` handles), so their
        // d_ptr is a link-time vaddr the loader biases by the load base only at
        // use. Bias it here and bounds-check the result within a loadable range.
        if !vaddr_in_load(&self.loads, vaddr, 1) {
            return Err(Error::Malformed(
                "version-table vaddr outside loadable segments",
            ));
        }
        (self.base as u64)
            .checked_add(vaddr)
            .ok_or(Error::Malformed("version-table address overflow"))
    }
    fn read(&self, cursor: u64, buf: &mut [u8]) -> Result<()> {
        read_live(self.base, &self.loads, cursor, buf)
    }
    fn slot_address(&self, r_offset: u64) -> Result<u64> {
        let addr = (self.base as u64)
            .checked_add(r_offset)
            .ok_or(Error::Malformed("slot address overflow"))?;
        // The GOT slot must lie within a loadable segment and be pointer-aligned:
        // the install/restore path reinterprets it as an `&AtomicUsize`, so a
        // misaligned cell would be instant UB.
        if !vaddr_in_load(&self.loads, r_offset, crate::arch::PTR_SIZE as u64) {
            return Err(Error::Malformed(
                "relocation offset outside loadable segments",
            ));
        }
        if !addr.is_multiple_of(core::mem::align_of::<usize>() as u64) {
            return Err(Error::Malformed("relocation slot is not pointer-aligned"));
        }
        Ok(addr)
    }
}

impl Module {
    /// Acquire a module from a `dlopen` handle. On glibc, `dlinfo`
    /// (`RTLD_DI_LINKMAP`) yields the object's load bias, which selects its
    /// `dl_iterate_phdr` entry.
    ///
    /// # Safety
    /// `handle` must be a live handle returned by `dlopen`/`dlmopen` for a
    /// currently-loaded object.
    pub unsafe fn from_handle(handle: *mut c_void) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::ModuleNotFound {
                reason: "null handle",
                os_error: 0,
            });
        }
        let mut lmap: *mut LinkMapHead = core::ptr::null_mut();
        // SAFETY: `handle` is a live dlopen handle; `lmap` is a valid out-pointer.
        let rc = unsafe {
            libc::dlinfo(
                handle,
                libc::RTLD_DI_LINKMAP,
                (&raw mut lmap).cast::<c_void>(),
            )
        };
        if rc != 0 || lmap.is_null() {
            return Err(Error::ModuleNotFound {
                reason: "dlinfo handle",
                os_error: 0,
            });
        }
        // SAFETY: `lmap` points at a live `link_map`; its first field is `l_addr`.
        let base = unsafe { (*lmap).l_addr };
        Self::from_base(base)
    }

    /// Acquire the module containing `address` — e.g. the address of an exported
    ///
    /// # Safety
    /// `address` must point into a loadable segment of a currently-mapped object.
    pub unsafe fn from_address(address: *const c_void) -> Result<Self> {
        let target = address as usize;
        let found = dlpi::for_each(|obj| obj.contains_address(target).then(|| obj.clone()));
        match found {
            Some(obj) => Self::from_raw(&obj),
            None => Err(Error::ModuleNotFound {
                reason: "address",
                os_error: 0,
            }),
        }
    }

    /// Acquire an already-loaded module by SONAME / path (`dlopen` with
    /// `RTLD_NOLOAD`, so a not-yet-loaded object is reported rather than loaded).
    pub fn from_name(name: &str) -> Result<Self> {
        let mut c_name: Vec<u8> = name.as_bytes().to_vec();
        c_name.push(0);
        // SAFETY: `c_name` is a valid NUL-terminated C string.
        let handle = unsafe {
            libc::dlopen(
                c_name.as_ptr().cast::<c_char>(),
                libc::RTLD_LAZY | libc::RTLD_NOLOAD,
            )
        };
        if handle.is_null() {
            return Err(Error::ModuleNotFound {
                reason: "name",
                os_error: 0,
            });
        }
        // SAFETY: `handle` is a live dlopen handle we own.
        let module = unsafe { Self::from_handle(handle) };
        // SAFETY: closing the handle we just opened; the object stays mapped
        // because `RTLD_NOLOAD` did not add a fresh reference to an already-loaded
        // object beyond the one this call holds.
        unsafe { libc::dlclose(handle) };
        module
    }

    /// Enumerate every object loaded in the process (`Scope::AllModules`). Objects
    /// that fail validation (no `PT_DYNAMIC`, unsupported machine) are skipped.
    pub fn enumerate() -> Result<Vec<Self>> {
        let mut out = Vec::new();
        dlpi::for_each(|obj| -> Option<()> {
            if let Ok(module) = Self::from_raw(obj) {
                out.push(module);
            }
            None
        });
        Ok(out)
    }

    /// Build and validate a module from an object's load bias.
    fn from_base(base: usize) -> Result<Self> {
        let obj = dlpi::find_base(base).ok_or(Error::ModuleNotFound {
            reason: "base",
            os_error: 0,
        })?;
        Self::from_raw(&obj)
    }

    /// Validate and capture identity from a raw `dl_iterate_phdr` object.
    fn from_raw(obj: &Object) -> Result<Self> {
        let ph = elf::parse_program_headers(&obj.phdrs, obj.phnum)?;
        let dyn_vaddr = ph
            .dynamic_vaddr
            .ok_or(Error::Malformed("object has no PT_DYNAMIC"))?;
        let bias = obj.base as u64;

        // Validate the ELF header, read from the object's mapped image.
        let ehdr_addr = bias
            .checked_add(ph.ehdr_vaddr)
            .ok_or(Error::Malformed("ELF header address overflow"))?;
        let mut ehdr = [0u8; 64];
        read_live(obj.base, &ph.loads, ehdr_addr, &mut ehdr)?;
        let machine = elf::validate_ehdr(&ehdr)?;

        let dynamic_addr = bias
            .checked_add(dyn_vaddr)
            .ok_or(Error::Malformed("dynamic table address overflow"))?;

        let (lo, hi) = span_of(obj.base, &ph.loads);
        Ok(Self {
            base: obj.base,
            loads: ph.loads,
            machine,
            dynamic_addr,
            span: (lo, hi),
            path: obj.name.clone(),
        })
    }

    /// File name of the module (final path component).
    #[must_use]
    pub fn name(&self) -> &str {
        match self.path.rfind('/') {
            Some(pos) => self.path.get(pos + 1..).unwrap_or(&self.path),
            None => &self.path,
        }
    }

    /// Full path of the module (`dlpi_name`; empty for the main executable).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Load bias (`dlpi_addr`) of the mapped image.
    #[must_use]
    pub fn base(&self) -> *const c_void {
        self.base as *const c_void
    }

    /// Size of the mapped image span (`hi - lo` over its loadable segments).
    #[must_use]
    pub fn size(&self) -> usize {
        self.span.1.saturating_sub(self.span.0)
    }

    pub(crate) fn base_addr(&self) -> usize {
        self.base
    }

    /// Enumerate the module's import slots (`JUMP_SLOT` + `GLOB_DAT`).
    ///
    /// Every returned slot's address was validated to lie inside a loadable
    /// segment and to be pointer-aligned, so it is safe to pass to
    /// [`crate::install`].
    pub fn imports(&self) -> Result<Vec<ImportSlot>> {
        Ok(elf::parse_imports(self)?
            .into_iter()
            .map(|raw| ImportSlot::from_raw(self.base, raw))
            .collect())
    }
}

/// Lowest / highest mapped absolute address over the loadable segments.
fn span_of(base: usize, loads: &[Segment]) -> (usize, usize) {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for seg in loads {
        // ELF64 addresses on the 64-bit-only targets fit `usize`; a value that did
        // not would simply saturate this informational span.
        let vaddr = usize::try_from(seg.vaddr).unwrap_or(usize::MAX);
        let memsz = usize::try_from(seg.memsz).unwrap_or(0);
        let start = base.wrapping_add(vaddr);
        let end = start.wrapping_add(memsz);
        lo = lo.min(start);
        hi = hi.max(end);
    }
    if lo == usize::MAX {
        (base, base)
    } else {
        (lo, hi)
    }
}
