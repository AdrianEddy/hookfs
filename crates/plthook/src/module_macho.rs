//! Live Mach-O module acquisition and import enumeration.
//!
//! The backend reads mapped metadata from memory and reads chained-fixup encodings
//! from the backing file when one is available. A dyld shared-cache image has no
//! readable backing file, so its in-memory indirect-symbol imports are used instead.
//! The same implementation serves macOS and iOS/iPadOS.

use crate::error::{Error, Result};
use crate::macho::{self, MachOView, Segment};
use crate::slot::ImportSlot;
use core::ffi::{c_char, c_int, c_void};
use mach2::dyld;
use mach2::kern_return::KERN_SUCCESS;
use mach2::traps::mach_task_self;
use mach2::vm::mach_vm_region;
use mach2::vm_region::{VM_REGION_BASIC_INFO_64, vm_region_basic_info_data_64_t};
use std::sync::{Arc, OnceLock};

/// A validated, currently-loaded Mach-O image.
#[derive(Debug, Clone)]
pub struct Module {
    /// Header address A?€�t also the image's base (the `mach_header_64` sits at the
    /// start of `__TEXT`, which maps at `slide + text_vmaddr`).
    base: usize,
    /// vmaddr slide (`_dyld_get_image_vmaddr_slide`).
    slide: isize,
    /// `LC_SEGMENT_64` table, for file-offset A?†�t address translation and bounds.
    segments: Vec<Segment>,
    /// `cputype` (picks the supported-architecture check).
    cpu_type: u32,
    /// `cpusubtype` (its low bits mark arm64e).
    cpu_subtype: u32,
    /// Lowest / highest mapped byte (`[lo, hi)`), for the reported size.
    span: (usize, usize),
    /// The image path (`_dyld_get_image_name`).
    path: String,
    /// The original on-disk bytes of the image's **thin slice**, for the
    /// chained-fixup chain walk. Loaded lazily on first demand and cached: `None`
    /// once loading is attempted and the file is unreadable (a dyld-shared-cache
    /// library) or carries no slice for the loaded architecture. A universal file is
    /// narrowed to the loaded arch's slice here, so chain file offsets (relative to
    /// the thin header) address the correct bytes. See [`Module::thin_slice`].
    thin_slice: OnceLock<Option<Arc<[u8]>>>,
}

/// Copy `buf.len()` bytes from live process memory at absolute address `addr`.
///
/// # Safety
/// `[addr, addr+buf.len())` must lie within a mapped, readable region of this
/// process (the caller validates it against a segment's mapped range first).
unsafe fn read_mem(addr: usize, buf: &mut [u8]) {
    // SAFETY: caller guarantees the source range is mapped and readable; `buf`
    // cannot overlap the foreign image.
    unsafe {
        core::ptr::copy_nonoverlapping(addr as *const u8, buf.as_mut_ptr(), buf.len());
    }
}

/// The segment whose on-disk range `[fileoff, fileoff+filesize)` contains file
/// offset `off`, and the absolute address `off` maps to in memory.
fn locate(segments: &[Segment], slide: isize, off: u64, len: u64) -> Result<usize> {
    for seg in segments {
        if off >= seg.fileoff && off < seg.fileoff.saturating_add(seg.filesize) {
            // The read must stay within the segment's file-backed range.
            if off
                .checked_add(len)
                .is_none_or(|end| end > seg.fileoff.saturating_add(seg.filesize))
            {
                return Err(Error::Malformed("read spans past the segment"));
            }
            let vaddr = seg
                .vmadd
                .checked_add(off - seg.fileoff)
                .ok_or(Error::Malformed("segment vaddr overflow"))?;
            let addr = usize::try_from(vaddr).map_err(|_| Error::Malformed("address overflow"))?;
            return Ok(addr.wrapping_add_signed(slide));
        }
    }
    Err(Error::Malformed("file offset outside any segment"))
}

impl MachOView for Module {
    fn cpu_type(&self) -> u32 {
        self.cpu_type
    }
    fn cpu_subtype(&self) -> u32 {
        self.cpu_subtype
    }
    fn read(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        let addr = locate(&self.segments, self.slide, off, buf.len() as u64)?;
        // SAFETY: `locate` validated `[addr, addr+len)` lies within a mapped,
        // file-backed segment range of this live image.
        unsafe { read_mem(addr, buf) };
        Ok(())
    }
    fn file_backed(&self) -> bool {
        self.thin_slice().is_some()
    }
    fn read_slot_encoding(&self, off: u64) -> Result<u64> {
        // The live in-memory slot has been fixed up by dyld; read the original chain
        // encoding from the on-disk file instead. `off` is a thin-slice-relative file
        // offset, and `thin_slice` holds exactly that slice (a fat container is
        // already narrowed to the loaded arch), so it indexes directly.
        let bytes = self
            .thin_slice()
            .ok_or(Error::Malformed("no on-disk file bytes"))?;
        let start = usize::try_from(off).map_err(|_| Error::Malformed("offset overflow"))?;
        let end = start
            .checked_add(8)
            .ok_or(Error::Malformed("read overflow"))?;
        let slice = bytes
            .get(start..end)
            .ok_or(Error::Malformed("chain read past file end"))?;
        let arr: [u8; 8] = slice
            .try_into()
            .map_err(|_| Error::Malformed("chain read"))?;
        Ok(u64::from_le_bytes(arr))
    }
    fn slot_address(&self, off: u64) -> Result<u64> {
        let addr = locate(
            &self.segments,
            self.slide,
            off,
            crate::arch::PTR_SIZE as u64,
        )?;
        // The slot is reinterpreted as an `&AtomicUsize`, so a misaligned cell would
        // be instant UB.
        if !addr.is_multiple_of(core::mem::align_of::<usize>()) {
            return Err(Error::Malformed("slot is not pointer-aligned"));
        }
        Ok(addr as u64)
    }
}

/// A bootstrap view that reads only the header + load-command region of a live
/// image, so the segment table can be recovered before the full [`Module`] view
/// exists. The `mach_header_64` sits at `__TEXT` (`fileoff 0`, mapping at `base`), so
/// a header/command file offset `off` is simply the memory address `base + off`.
struct HeaderView {
    base: usize,
    /// Number of bytes readable from `base` A?€�t the extent of the mapped VM region the
    /// header lives in (`mach_vm_region`), so a corrupt `sizeofcmds` cannot drive a
    /// load-command read past the actual mapping. See [`mapped_extent_from`].
    limit: u64,
}

/// Fallback upper bound on a header/load-command read, used only when the mapped
/// extent cannot be queried (`mach_vm_region` failed). A corrupt-but-mapped heade
/// then still cannot walk more than this before [`crate::macho::read_load_commands`]
/// rejects it, and no genuine load-command region approaches it.
const MAX_HEADER_SPAN: u64 = 64 << 20;

/// Number of bytes mapped and readable starting at `addr`, from the enclosing VM
/// region (`mach_vm_region`) A?€�t i.e. the extent of the segment the image header lives
/// in (`__TEXT` starts at `base`, so this is its mapped size). Returns
/// [`MAX_HEADER_SPAN`] if the region cannot be queried or does not contain `addr`, so
/// a caller always has a finite, sound read bound.
fn mapped_extent_from(addr: usize) -> u64 {
    // SAFETY: reads the (static) task-self port; no preconditions.
    let task = unsafe { mach_task_self() };
    let mut region_addr = addr as u64;
    let mut region_size: u64 = 0;
    // SAFETY: `vm_region_basic_info_64` is plain-old-data; all-zero is valid.
    let mut info: vm_region_basic_info_data_64_t = unsafe { core::mem::zeroed() };
    let mut count = vm_region_basic_info_data_64_t::count();
    let mut object_name: mach2::port::mach_port_t = 0;
    // `mach_vm_region` reports the region containing `region_addr`, or the next one
    // above it, updating `region_addr`/`region_size` to that region's bounds.
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
    let target = addr as u64;
    // Accept only when the reported region actually covers `addr`; otherwise `addr`
    // fell in a hole before the next region and no sound extent is known.
    if kr == KERN_SUCCESS
        && region_addr <= target
        && let Some(region_end) = region_addr.checked_add(region_size)
        && target < region_end
    {
        region_end - target
    } else {
        MAX_HEADER_SPAN
    }
}

impl MachOView for HeaderView {
    fn cpu_type(&self) -> u32 {
        // Read once from the header (offset 4); only used defensively.
        let mut b = [0u8; 4];
        // SAFETY: a live mach header's first 8 bytes are always readable.
        unsafe { read_mem(self.base + 4, &mut b) };
        u32::from_le_bytes(b)
    }
    fn cpu_subtype(&self) -> u32 {
        let mut b = [0u8; 4];
        // SAFETY: a live mach header's first 12 bytes are always readable.
        unsafe { read_mem(self.base + 8, &mut b) };
        u32::from_le_bytes(b)
    }
    fn read(&self, off: u64, buf: &mut [u8]) -> Result<()> {
        let end = off
            .checked_add(buf.len() as u64)
            .ok_or(Error::Malformed("read overflow"))?;
        // Bound by the header segment's actual mapped extent, so a corrupt-but-mapped
        // header claiming a huge `sizeofcmds` can never drive this read past the
        // mapping (A?†’ SIGSEGV).
        if end > self.limit {
            return Err(Error::Malformed("header read out of range"));
        }
        let addr = self
            .base
            .checked_add(usize::try_from(off).unwrap_or(usize::MAX))
            .ok_or(Error::Malformed("header address overflow"))?;
        // SAFETY: the header + load commands lie in `__TEXT` (fileoff 0) mapping at
        // `base`; `end <= self.limit` keeps `[base+off, base+off+len)` inside the VM
        // region that starts at `base`, so the range is mapped and readable.
        unsafe { read_mem(addr, buf) };
        Ok(())
    }
    fn file_backed(&self) -> bool {
        false
    }
    fn read_slot_encoding(&self, _off: u64) -> Result<u64> {
        Err(Error::Malformed("bootstrap view has no slot encodings"))
    }
    fn slot_address(&self, _off: u64) -> Result<u64> {
        Err(Error::Malformed("bootstrap view resolves no slots"))
    }
}

impl Module {
    /// Acquire the module containing `address` A?€�t e.g. the address of an exported
    /// image list.
    ///
    /// # Safety
    /// `address` must point into a currently-mapped image.
    pub unsafe fn from_address(address: *const c_void) -> Result<Self> {
        // SAFETY: `Dl_info` is plain-old-data; `dladdr` fills it. `address` is only a
        // lookup key.
        let mut info: libc::Dl_info = unsafe { core::mem::zeroed() };
        // SAFETY: `info` is a valid out-pointer.
        if unsafe { libc::dladdr(address, &raw mut info) } == 0 {
            return Err(Error::ModuleNotFound {
                reason: "address",
                os_error: 0,
            });
        }
        let base = info.dli_fbase as usize;
        Self::from_base(base)
    }

    /// Acquire an already-loaded module by path / trailing name (matching the
    /// `plthook_osx` open-by-name rule: a non-absolute name matches any image whose
    /// path ends with it).
    pub fn from_name(name: &str) -> Result<Self> {
        // SAFETY: dyld image accessors have no preconditions.
        let count = unsafe { dyld::_dyld_image_count() };
        for i in 0..count {
            let Some(image_name) = image_name(i) else {
                continue;
            };
            let matches = if name.starts_with('/') {
                image_name == name
            } else {
                image_name.ends_with(name)
            };
            if matches {
                return Self::from_index(i);
            }
        }
        Err(Error::ModuleNotFound {
            reason: "name",
            os_error: 0,
        })
    }

    /// Acquire a module from a `dlopen` handle. macOS exposes no direct handle A?†’
    /// header mapping, so A?€�t like `plthook_osx` A?€�t each image is re-opened with
    /// `RTLD_NOLOAD` and its handle compared to `handle`.
    ///
    /// macOS *tags* the returned handle with `RTLD_FIRST`, so `dlopen(path,
    /// RTLD_LAZY)` and `dlopen(path, RTLD_LAZY | RTLD_FIRST)` return **different**
    /// handle values for the very same image. The caller's `handle` came from one
    /// form or the other, so A?€�t exactly as `plthook_osx` does A?€�t this probes **both**
    /// flag sets and matches either; probing only the `RTLD_FIRST` variant (as an
    /// earlier version did) silently failed to resolve the common
    /// `dlopen(path, RTLD_LAZY)` handle. Image 0 (the main executable) is probed via
    /// `dlopen(NULL, A?€¦)`, whose handle differs from opening the executable by path.
    ///
    /// # Safety
    /// `handle` must be a live handle returned by `dlopen` for a loaded image.
    pub unsafe fn from_handle(handle: *mut c_void) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::ModuleNotFound {
                reason: "null handle",
                os_error: 0,
            });
        }
        // SAFETY: dyld image accessors have no preconditions.
        let count = unsafe { dyld::_dyld_image_count() };
        for extra in [0, libc::RTLD_FIRST] {
            let flags = libc::RTLD_LAZY | libc::RTLD_NOLOAD | extra;
            for i in 0..count {
                // The main executable's handle is the one `dlopen(NULL, A?€¦)` returns;
                // opening it by path yields a different handle. Named images use
                // their path, NUL-terminated for the C ABI.
                let c_name: Option<Vec<u8>> = if i == 0 {
                    None
                } else {
                    let Some(name) = image_name(i) else { continue };
                    let mut bytes: Vec<u8> = name.into_bytes();
                    bytes.push(0);
                    Some(bytes)
                };
                let path_ptr = match &c_name {
                    None => core::ptr::null(),
                    Some(bytes) => bytes.as_ptr().cast::<c_char>(),
                };
                // SAFETY: `path_ptr` is NULL (main program) or a valid NUL-terminated
                // path; `RTLD_NOLOAD` reports an already-loaded image without loading.
                let probe = unsafe { libc::dlopen(path_ptr, flags) };
                if probe.is_null() {
                    continue;
                }
                // SAFETY: `probe` is a handle we own; releasing our reference.
                unsafe { libc::dlclose(probe) };
                if probe == handle {
                    return Self::from_index(i);
                }
            }
        }
        Err(Error::ModuleNotFound {
            reason: "handle",
            os_error: 0,
        })
    }

    /// Enumerate every image loaded in the process (`Scope::AllModules`). Images that
    /// fail validation (unsupported cputype, unreadable header) are skipped.
    pub fn enumerate() -> Result<Vec<Self>> {
        // SAFETY: dyld image accessors have no preconditions.
        let count = unsafe { dyld::_dyld_image_count() };
        let mut out = Vec::new();
        for i in 0..count {
            if let Ok(module) = Self::from_index(i) {
                out.push(module);
            }
        }
        Ok(out)
    }

    /// Build a module from the header address of a loaded image.
    fn from_base(base: usize) -> Result<Self> {
        // SAFETY: dyld image accessors have no preconditions.
        let count = unsafe { dyld::_dyld_image_count() };
        for i in 0..count {
            // SAFETY: `i < count`.
            if unsafe { dyld::_dyld_get_image_header(i) } as usize == base {
                return Self::from_index(i);
            }
        }
        Err(Error::ModuleNotFound {
            reason: "base",
            os_error: 0,
        })
    }

    /// Validate and capture identity from a dyld image index.
    fn from_index(idx: u32) -> Result<Self> {
        // SAFETY: `idx < _dyld_image_count()` at the call sites.
        let header = unsafe { dyld::_dyld_get_image_header(idx) } as usize;
        if header == 0 {
            return Err(Error::ModuleNotFound {
                reason: "index",
                os_error: 0,
            });
        }
        // SAFETY: `idx` is valid.
        let slide = unsafe { dyld::_dyld_get_image_vmaddr_slide(idx) };
        let path = image_name(idx).unwrap_or_default();

        // Validate the header magic before trusting any parsed field.
        let mut magic = [0u8; 4];
        // SAFETY: a live mach header's first 4 bytes are always readable.
        unsafe { read_mem(header, &mut magic) };
        if u32::from_le_bytes(magic) != macho::MH_MAGIC_64 {
            return Err(Error::Unsupported(
                "image is not a little-endian 64-bit Mach-O",
            ));
        }

        // Bootstrap: read the load commands to recover the segment table. Reads are
        // bounded to the header segment's mapped VM extent (m5), so a corrupt heade
        // cannot walk off the mapping.
        let boot = HeaderView {
            base: header,
            limit: mapped_extent_from(header),
        };
        if !crate::arch::macho_cpu_supported(boot.cpu_type()) {
            return Err(Error::Unsupported("Mach-O cputype is not x86-64 or arm64"));
        }
        let segments = macho::segments_of(&boot)?;
        if segments.is_empty() {
            return Err(Error::Malformed("image has no LC_SEGMENT_64"));
        }

        let span = span_of(&segments, slide)?;

        // The on-disk file is NOT read here: it is only needed for the chained-fixup
        // chain walk, and then only for images that carry `LC_DYLD_CHAINED_FIXUPS`, so
        // it is loaded lazily by `thin_slice` on first demand (m1). This keeps
        // `Scope::AllModules` from eagerly reading A?€�t and retaining A?€�t every dylib.
        Ok(Self {
            base: header,
            slide,
            segments,
            cpu_type: boot.cpu_type(),
            cpu_subtype: boot.cpu_subtype(),
            span,
            path,
            thin_slice: OnceLock::new(),
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

    /// Full path of the module (`_dyld_get_image_name`).
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Base (Mach-O header) address of the mapped image.
    #[must_use]
    pub fn base(&self) -> *const c_void {
        self.base as *const c_void
    }

    /// Size of the mapped image span (`hi - lo` over its segments).
    #[must_use]
    pub fn size(&self) -> usize {
        self.span.1.saturating_sub(self.span.0)
    }

    pub(crate) fn base_addr(&self) -> usize {
        self.base
    }

    /// The original on-disk bytes of this image's **thin slice**, loaded lazily on
    /// first call and cached for the module's lifetime. Only the chained-fixup chain
    /// walk needs them (the in-memory encodings are overwritten by dyld), so the file
    /// is untouched for any image without `LC_DYLD_CHAINED_FIXUPS` (m1). A universal
    /// ("fat") file is narrowed to the loaded architecture's slice, so a chain's
    /// thin-slice-relative file offsets index the correct bytes (M2). `None` if the
    /// file is unreadable (a dyld-shared-cache library) or carries no matching slice.
    fn thin_slice(&self) -> Option<&[u8]> {
        self.thin_slice
            .get_or_init(|| load_thin_slice(&self.path, self.cpu_type, self.cpu_subtype))
            .as_deref()
    }

    /// Enumerate the module's rebindable import slots (indirect-symbol,
    ///
    /// Every returned slot's address was validated to lie inside a segment and to be
    /// pointer-aligned, so it is safe to pass to [`crate::install`]. Authenticated
    /// (arm64e) slots are flagged and refused by the transaction, not written.
    pub fn imports(&self) -> Result<Vec<ImportSlot>> {
        Ok(macho::parse_imports(self)?
            .into_iter()
            .map(|raw| ImportSlot::from_raw(self.base, raw))
            .collect())
    }
}

/// The image path at dyld index `idx`, if any.
fn image_name(idx: u32) -> Option<String> {
    // SAFETY: `idx < _dyld_image_count()` at the call sites.
    let ptr = unsafe { dyld::_dyld_get_image_name(idx) };
    if ptr.is_null() {
        return None;
    }
    // SAFETY: `_dyld_get_image_name` returns a NUL-terminated C string owned by dyld.
    Some(
        unsafe { core::ffi::CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Lowest / highest mapped absolute address over the segments (`slide + vmaddr`).
fn span_of(segments: &[Segment], slide: isize) -> Result<(usize, usize)> {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for seg in segments {
        let vmaddr =
            usize::try_from(seg.vmaddr).map_err(|_| Error::Malformed("vmaddr overflow"))?;
        let vmsize = usize::try_from(seg.vmsize).unwrap_or(0);
        let start = vmaddr.wrapping_add_signed(slide);
        let end = start.wrapping_add(vmsize);
        lo = lo.min(start);
        hi = hi.max(end);
    }
    Ok(if lo == usize::MAX { (0, 0) } else { (lo, hi) })
}

// ---- Universal ("fat") on-disk container: thin-slice selection (M2) ---------
//
// A loaded image's IN-MEMORY header is always a thin `mach_header_64` A?€�t dyld maps one
// slice A?€�t but its ORIGINAL on-disk file (read here for the chained-fixup chain
// encodings) may be a UNIVERSAL ("fat") container that holds several architecture
// slices, each at a nonzero `fat_arch.offset`. A fixup chain's file offsets are
// relative to the *thin* header (offset 0 of the slice), so the file must be narrowed
// to the loaded arch's slice before those offsets are used A?€�t otherwise every
// `read_slot_encoding(off)` indexes `file[off]` instead of `file[slice_off + off]`
// and decodes garbage. `plthook_osx` seeks the raw file offset and has this bug; the
// shared, host-tested [`macho::thin_slice_range`] selects the matching slice instead
// (`<mach-o/fat.h>`). That selector A?€�t and its fat-fixture tests A?€�t live in
// [`crate::macho`], so there is a single implementation exercised on any host.

/// Read the original on-disk file at `path` and return the bytes of the **thin
/// slice** for the loaded image's `(cpu_type, cpu_subtype)`. Only that slice is
/// retained A?€�t a universal container is never kept whole (m1). `None` if the file is
/// unreadable (a dyld-shared-cache library A?€�t the norm for iOS system libraries) o
/// carries no slice for that architecture, in which case the chained-fixup walk is
/// skipped and the indirect-symbol path carries the enumeration. The fat-slice
fn load_thin_slice(path: &str, cpu_type: u32, cpu_subtype: u32) -> Option<Arc<[u8]>> {
    let file = std::fs::read(path).ok()?;
    let range = macho::thin_slice_range(&file, cpu_type, cpu_subtype).ok()?;
    Some(Arc::from(file.get(range)?))
}
