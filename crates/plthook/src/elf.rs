//!
//! This module decodes the dynamic-linking metadata of an ELF64 image and yields
//! one [`RawImport`] per rebindable GOT/PLT slot. It uses no third-party object
//! count, string, relocation, and slot address is validated against the image's
//! loadable ranges *before* the byte is read, so a malformed image yields a typed
//! [`Error`] and never a read out of bounds.
//!
//! # Host-independent by construction
//! The parser never touches process memory or a syscall directly. It works
//! entirely through the [`ElfView`] trait, which abstracts the *addressing mode*
//! of the two backings the engine parses:
//!
//! * **Live memory** (the Linux runtime, [`crate::module`]): the dynamic-linker
//!   has already relocated the *address* dynamic-section pointers
//!   (`DT_SYMTAB`/`DT_STRTAB`/`DT_JMPREL`/`DT_RELA`/`DT_VERSYM`) to absolute
//!   addresses in place (glibc `!DL_RO_DYN_SECTION`, true on x86-64 and
//!   `AArch64`), so such a `d_ptr` *is* the address to read; a slot address is
//!   `load_bias + r_offset`. The one exception is `DT_VERNEED`/`DT_VERDEF`, which
//!   glibc's `ADJUST_DYN_INFO` deliberately leaves as a link-time vaddr — those go
//!   through [`ElfView::vaddr_cursor`], which adds the load bias.
//! * **On-disk file bytes** (the host-independent parser tests): the dynamic
//!   pointers are still link-time virtual addresses, translated to file offsets
//!   through the `PT_LOAD` program headers.
//!
//! Because the two modes differ *only* in that addressing, the format logic here
//! is shared verbatim and is exercised on this Windows host against the real
//! Blackmagic `.so` files (see the tests).

use crate::arch;
use crate::error::{Error, Result};
use crate::import::RawImport;
use crate::{ImportKind, Symbol};
use std::collections::HashMap;
use std::sync::Arc;

// ---- ELF constants ---------------------------------------------------------

/// ELF identification: 64-bit class.
const ELFCLASS64: u8 = 2;
/// ELF identification: two's-complement little-endian data.
const ELFDATA2LSB: u8 = 1;
/// `e_ident` index of the class byte.
const EI_CLASS: usize = 4;
/// `e_ident` index of the data-encoding byte.
const EI_DATA: usize = 5;
/// `e_type`: an executable file.
const ET_EXEC: u16 = 2;
/// `e_type`: a shared object (`.so` / PIE).
const ET_DYN: u16 = 3;

/// Program-header type: a loadable segment.
const PT_LOAD: u32 = 1;
/// Program-header type: the dynamic-linking table.
const PT_DYNAMIC: u32 = 2;

/// Size of one `Elf64_Phdr`.
pub(crate) const PHDR_SIZE: usize = 56;
/// Size of one `Elf64_Dyn`.
const DYN_SIZE: u64 = 16;
/// Size of one `Elf64_Sym`.
const SYM_SIZE: u64 = 24;
/// Size of one `Elf64_Rela` (x86-64 / `AArch64` both use RELA).
const RELA_SIZE: u64 = 24;

// Dynamic-array tags (`d_tag`).
const DT_NULL: i64 = 0;
const DT_PLTRELSZ: i64 = 2;
const DT_STRTAB: i64 = 5;
const DT_SYMTAB: i64 = 6;
const DT_RELA: i64 = 7;
const DT_RELASZ: i64 = 8;
const DT_RELAENT: i64 = 9;
const DT_STRSZ: i64 = 10;
const DT_SYMENT: i64 = 11;
const DT_PLTREL: i64 = 20;
const DT_JMPREL: i64 = 23;
const DT_VERSYM: i64 = 0x6fff_fff0;
const DT_VERNEED: i64 = 0x6fff_fffe;
const DT_VERNEEDNUM: i64 = 0x6fff_ffff;

/// `DT_PLTREL` value meaning the PLT relocations use the `Elf64_Rela` form.
const DT_PLTREL_IS_RELA: u64 = DT_RELA as u64;

/// The `versym` version-index mask (bit 15 is the "hidden" flag).
const VERSYM_VERSION_MASK: u16 = 0x7fff;
/// Version indices 0 (`*_LOCAL`) and 1 (`*_GLOBAL`) carry no named version.
const VER_NDX_GLOBAL: u16 = 1;

/// Defensive iteration bounds. A conformant image stays far below these; exceeding
/// one means the tables are malformed (or maliciously unterminated), so the parser
/// stops with [`Error::Malformed`] instead of looping.
const MAX_DYN: usize = 4096;
/// Upper bound on relocations in one table (`.rela.plt` or `.rela.dyn`). Set far
/// above any real image — a ~400 MB `.rela.dyn` — so a conformant library never
/// trips it, while a corrupt `DT_RELASZ`/`DT_PLTRELSZ` cannot drive an unbounded
/// loop. Exceeding it is reported as [`Error::Malformed`] rather than **silently
/// truncating** the walk (which would drop the `GLOB_DAT` imports that `-z
/// combreloc` sorts to the tail of `.rela.dyn`, making them unhookable with no
/// diagnostic).
const MAX_RELOCS: u64 = 1 << 24;
const MAX_VERNEED: usize = 4096;
const MAX_VERNAUX: usize = 4096;
const MAX_NAME_LEN: usize = 8192;

// ---- The addressing-mode abstraction ---------------------------------------

/// One loadable segment of an ELF image, enough to translate a virtual address to
/// a read cursor and to bounds-check a slot in either addressing mode.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Segment {
    /// Link-time virtual address of the segment start (`p_vaddr`).
    pub(crate) vaddr: u64,
    /// In-memory size (`p_memsz`) — the bound for a live read (covers `.bss`).
    /// Read only by the live Linux addressing mode.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) memsz: u64,
    /// File offset of the segment start (`p_offset`). Read only by the offline
    /// (file-bytes) addressing mode — i.e. the host-independent parser tests; the
    /// live Linux runtime reads at absolute addresses and never needs it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) offset: u64,
    /// On-disk size (`p_filesz`) — the offline addressing mode's bound (see
    /// `offset`).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) filesz: u64,
}

/// The dynamic-linking metadata the parser reads, abstracted over its addressing
/// mode so the same logic serves both a live image and offline file bytes.
///
/// A *cursor* is an opaque `u64` produced by [`Self::dynamic_cursor`] /
/// [`Self::dynptr_cursor`] and consumed by [`Self::read`]; the implementation
/// decides whether it denotes an absolute address (live) or a file offset
/// (on-disk). The parser only ever adds byte offsets to a cursor, which is valid
/// in both interpretations.
pub(crate) trait ElfView {
    /// The image's `e_machine` (used to pick the relocation constants).
    fn machine(&self) -> u16;

    /// Cursor at which the dynamic (`PT_DYNAMIC`) array begins.
    fn dynamic_cursor(&self) -> u64;

    /// Cursor at which a table located by a dynamic-entry pointer (`d_ptr`)
    /// begins. Fails if the pointer does not fall within a loadable range.
    ///
    /// Used for the dynamic pointers the loader relocates in place —
    /// `DT_SYMTAB`, `DT_STRTAB`, `DT_JMPREL`, `DT_RELA`, `DT_VERSYM`. On a live
    /// image these are already absolute addresses; on-disk they are link-time
    /// vaddrs. Contrast [`Self::vaddr_cursor`].
    fn dynptr_cursor(&self, d_ptr: u64) -> Result<u64>;

    /// Cursor at which a table located by a **link-time virtual address the loader
    /// does not relocate in place** begins — `DT_VERNEED` (and `DT_VERDEF`).
    ///
    /// glibc's `elf_get_dynamic_info` (`ADJUST_DYN_INFO`) rewrites the *address*
    /// dynamic pointers (`DT_SYMTAB`, `DT_STRTAB`, `DT_JMPREL`, `DT_RELA`,
    /// `DT_VERSYM`, …) to absolute addresses in place, but deliberately does **not**
    /// adjust `DT_VERNEED`/`DT_VERDEF`; it biases those by the load base only when
    /// it dereferences them (`_dl_check_map_versions`). So on a *live* image a
    /// `DT_VERNEED` `d_ptr` is still a link-time vaddr that needs the load bias
    /// added, whereas [`Self::dynptr_cursor`] receives an already-absolute pointer.
    /// On *on-disk* bytes every dynamic pointer is a link-time vaddr, so the two
    /// resolve identically there. Fails if the address is not within a loadable
    /// range.
    fn vaddr_cursor(&self, vaddr: u64) -> Result<u64>;

    /// Read exactly `buf.len()` bytes at `cursor`, bounds-checked against the
    /// image's loadable ranges. Fails ([`Error::Malformed`]) for any read that
    /// would leave those ranges.
    fn read(&self, cursor: u64, buf: &mut [u8]) -> Result<()>;

    /// Absolute patch-target address of a relocation whose offset is `r_offset`
    /// (`load_bias + r_offset` for a live image). Validated to be a
    /// pointer-aligned cell inside a loadable segment.
    fn slot_address(&self, r_offset: u64) -> Result<u64>;
}

// ---- Little-endian primitive reads through a view --------------------------

fn read_u16(view: &dyn ElfView, cursor: u64) -> Result<u16> {
    let mut b = [0u8; 2];
    view.read(cursor, &mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(view: &dyn ElfView, cursor: u64) -> Result<u32> {
    let mut b = [0u8; 4];
    view.read(cursor, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(view: &dyn ElfView, cursor: u64) -> Result<u64> {
    let mut b = [0u8; 8];
    view.read(cursor, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64(view: &dyn ElfView, cursor: u64) -> Result<i64> {
    let mut b = [0u8; 8];
    view.read(cursor, &mut b)?;
    Ok(i64::from_le_bytes(b))
}

/// Read a NUL-terminated string that lives at byte offset `off` into the string
/// table, whose cursor starts at `strtab` and whose size is `strsz`. The offset is
/// validated against `strsz`, the scan stops at the table end or [`MAX_NAME_LEN`],
/// and every byte read is additionally range-checked by the view.
fn read_str(view: &dyn ElfView, strtab: u64, strsz: u64, off: u32) -> Result<String> {
    let off = u64::from(off);
    if off >= strsz {
        return Err(Error::Malformed("string-table offset past DT_STRSZ"));
    }
    let max = (strsz - off).min(MAX_NAME_LEN as u64 + 1);
    let mut bytes = Vec::new();
    let mut i = 0u64;
    while i < max {
        let mut byte = [0u8; 1];
        view.read(strtab + off + i, &mut byte)?;
        if byte[0] == 0 {
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(byte[0]);
        i += 1;
    }
    Err(Error::Malformed("unterminated string-table entry"))
}

// ---- Little-endian reads over a borrowed byte slice ------------------------
//
// Used for header / program-header bytes the caller has already read into a
// buffer. Each returns a typed error rather than indexing, so a truncated slice
// never panics.

fn slice_le16(bytes: &[u8], off: usize) -> Result<u16> {
    let arr: [u8; 2] = bytes
        .get(off..off + 2)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::Malformed("read past slice bounds"))?;
    Ok(u16::from_le_bytes(arr))
}

fn slice_le32(bytes: &[u8], off: usize) -> Result<u32> {
    let arr: [u8; 4] = bytes
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::Malformed("read past slice bounds"))?;
    Ok(u32::from_le_bytes(arr))
}

fn slice_le64(bytes: &[u8], off: usize) -> Result<u64> {
    let arr: [u8; 8] = bytes
        .get(off..off + 8)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::Malformed("read past slice bounds"))?;
    Ok(u64::from_le_bytes(arr))
}

// ---- Shared header / program-header parsing --------------------------------

/// Result of walking the program headers: the loadable segments, the dynamic
/// table's virtual address, and the virtual address the ELF header maps at (the
/// `PT_LOAD` covering file offset 0).
#[derive(Debug)]
pub(crate) struct ProgramHeaders {
    pub(crate) loads: Vec<Segment>,
    pub(crate) dynamic_vaddr: Option<u64>,
    /// Virtual address the ELF header maps at (the `PT_LOAD` covering file offset
    /// 0). Read only by the live Linux mode to locate and validate the header; the
    /// offline mode reads the header at file offset 0 directly.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) ehdr_vaddr: u64,
}

/// Validate an ELF64 header (the first 64 bytes of the image) and return its
/// `e_machine`. Rejects a non-ELF, non-64-bit, big-endian, or non-object image
pub(crate) fn validate_ehdr(ehdr: &[u8]) -> Result<u16> {
    let head = ehdr
        .get(..20)
        .ok_or(Error::Malformed("truncated ELF header"))?;
    if head.get(..4) != Some(&[0x7f, b'E', b'L', b'F']) {
        return Err(Error::Malformed("missing ELF magic"));
    }
    if head.get(EI_CLASS) != Some(&ELFCLASS64) {
        return Err(Error::Unsupported("not an ELF64 image"));
    }
    if head.get(EI_DATA) != Some(&ELFDATA2LSB) {
        return Err(Error::Unsupported("not a little-endian ELF image"));
    }
    // e_type at offset 16, e_machine at offset 18 (both u16, LE).
    let e_type = slice_le16(ehdr, 16)?;
    let e_machine = slice_le16(ehdr, 18)?;
    if e_type != ET_EXEC && e_type != ET_DYN {
        return Err(Error::Malformed("ELF e_type is neither ET_EXEC nor ET_DYN"));
    }
    if arch::elf_reloc_kinds(e_machine).is_none() {
        return Err(Error::Unsupported(
            "ELF machine is not EM_X86_64 or EM_AARCH64",
        ));
    }
    Ok(e_machine)
}

/// Parse `phnum` program-header entries from the raw table bytes, collecting the
/// loadable segments and locating the dynamic table.
pub(crate) fn parse_program_headers(phdrs: &[u8], phnum: usize) -> Result<ProgramHeaders> {
    let mut loads = Vec::new();
    let mut dynamic_vaddr = None;
    let mut ehdr_vaddr = 0u64;
    for i in 0..phnum {
        let base = i
            .checked_mul(PHDR_SIZE)
            .ok_or(Error::Malformed("phdr offset overflow"))?;
        let entry = phdrs
            .get(base..base + PHDR_SIZE)
            .ok_or(Error::Malformed("program-header table truncated"))?;
        // Elf64_Phdr: p_type(0), p_flags(4), p_offset(8), p_vaddr(16), p_paddr(24),
        // p_filesz(32), p_memsz(40), p_align(48).
        let p_type = slice_le32(entry, 0)?;
        let p_offset = slice_le64(entry, 8)?;
        let p_vaddr = slice_le64(entry, 16)?;
        let p_filesz = slice_le64(entry, 32)?;
        let p_memsz = slice_le64(entry, 40)?;
        if p_type == PT_LOAD {
            loads.push(Segment {
                vaddr: p_vaddr,
                memsz: p_memsz,
                offset: p_offset,
                filesz: p_filesz,
            });
            if p_offset == 0 {
                ehdr_vaddr = p_vaddr;
            }
        } else if p_type == PT_DYNAMIC {
            dynamic_vaddr = Some(p_vaddr);
        }
    }
    if loads.is_empty() {
        return Err(Error::Malformed("no PT_LOAD segments"));
    }
    Ok(ProgramHeaders {
        loads,
        dynamic_vaddr,
        ehdr_vaddr,
    })
}

// The two addressing-mode conversions (link-time vaddr → file offset for offline
// bytes; vaddr-in-load bounds for live memory) live with their respective
// [`ElfView`] implementations — the live one in [`crate::module`], the offline one
// in the tests below — since each is used by exactly one mode.

// ---- The dynamic table -----------------------------------------------------

/// The dynamic-array entries the import walk needs.
#[derive(Debug, Default)]
struct DynamicInfo {
    symtab: Option<u64>,
    syment: Option<u64>,
    strtab: Option<u64>,
    strsz: Option<u64>,
    jmprel: Option<u64>,
    pltrelsz: Option<u64>,
    pltrel: Option<u64>,
    rela: Option<u64>,
    relasz: Option<u64>,
    relaent: Option<u64>,
    versym: Option<u64>,
    verneed: Option<u64>,
    verneednum: Option<u64>,
}

/// Walk the dynamic array to its `DT_NULL` terminator, collecting the tags the
/// import walk needs.
fn read_dynamic(view: &dyn ElfView) -> Result<DynamicInfo> {
    let mut info = DynamicInfo::default();
    let base = view.dynamic_cursor();
    for i in 0..MAX_DYN {
        let cursor = base + (i as u64) * DYN_SIZE;
        let tag = read_i64(view, cursor)?;
        if tag == DT_NULL {
            return Ok(info);
        }
        let val = read_u64(view, cursor + 8)?;
        match tag {
            DT_SYMTAB => info.symtab = Some(val),
            DT_SYMENT => info.syment = Some(val),
            DT_STRTAB => info.strtab = Some(val),
            DT_STRSZ => info.strsz = Some(val),
            DT_JMPREL => info.jmprel = Some(val),
            DT_PLTRELSZ => info.pltrelsz = Some(val),
            DT_PLTREL => info.pltrel = Some(val),
            DT_RELA => info.rela = Some(val),
            DT_RELASZ => info.relasz = Some(val),
            DT_RELAENT => info.relaent = Some(val),
            DT_VERSYM => info.versym = Some(val),
            DT_VERNEED => info.verneed = Some(val),
            DT_VERNEEDNUM => info.verneednum = Some(val),
            _ => {}
        }
    }
    Err(Error::Malformed("dynamic array is not DT_NULL-terminated"))
}

/// A resolved symbol version: the version string (e.g. `GLIBC_2.2.5`) and the
/// providing SONAME (e.g. `libc.so.6`), both from `DT_VERNEED`.
type VersionMap = HashMap<u16, (Arc<str>, Arc<str>)>;

/// Build the version-index → (version, SONAME) map from `DT_VERNEED`/`VERNAUX`.
// `vna_name` / `vna_next` are the ELF `Elf64_Vernaux` field names; their
// similarity is inherent to the format, not a smell.
#[allow(clippy::similar_names)]
fn build_version_map(
    view: &dyn ElfView,
    dynamic: &DynamicInfo,
    strtab: u64,
    strsz: u64,
) -> Result<VersionMap> {
    let mut map = VersionMap::new();
    let Some(verneed_ptr) = dynamic.verneed else {
        return Ok(map);
    };
    let count = dynamic.verneednum.map_or(MAX_VERNEED, |n| {
        usize::try_from(n).unwrap_or(MAX_VERNEED).min(MAX_VERNEED)
    });
    // `DT_VERNEED` is a link-time vaddr the loader does not relocate in place
    // (unlike `DT_VERSYM`), so it must go through `vaddr_cursor`, not
    // `dynptr_cursor` — otherwise a live image underflows on `d_ptr - base`.
    let mut need_cursor = view.vaddr_cursor(verneed_ptr)?;
    for _ in 0..count {
        // Elf64_Verneed: vn_version(0), vn_cnt(2), vn_file(4), vn_aux(8), vn_next(12).
        let vn_cnt = read_u16(view, need_cursor + 2)?;
        let vn_file = read_u32(view, need_cursor + 4)?;
        let vn_aux = read_u32(view, need_cursor + 8)?;
        let vn_next = read_u32(view, need_cursor + 12)?;
        let soname: Arc<str> = read_str(view, strtab, strsz, vn_file)?.into();

        let mut aux_cursor = need_cursor + u64::from(vn_aux);
        for _ in 0..(usize::from(vn_cnt)).min(MAX_VERNAUX) {
            // Elf64_Vernaux: vna_hash(0), vna_flags(4), vna_other(6), vna_name(8),
            // vna_next(12).
            let vna_other = read_u16(view, aux_cursor + 6)?;
            let vna_name = read_u32(view, aux_cursor + 8)?;
            let vna_next = read_u32(view, aux_cursor + 12)?;
            let version: Arc<str> = read_str(view, strtab, strsz, vna_name)?.into();
            map.insert(vna_other & VERSYM_VERSION_MASK, (version, soname.clone()));
            if vna_next == 0 {
                break;
            }
            aux_cursor += u64::from(vna_next);
        }
        // Each verneed entry advances by its own byte stride `vn_next` (not a fixed
        // struct size); a zero stride terminates the list.
        if vn_next == 0 {
            break;
        }
        need_cursor += u64::from(vn_next);
    }
    Ok(map)
}

/// Look up the version + providing SONAME of dynamic-symbol index `sym_index`
/// through `DT_VERSYM` and the verneed map. Returns `(None, "")` for an
/// unversioned import.
///
/// `versym_cursor` is the `DT_VERSYM` table base, resolved and validated **once**
/// by [`parse_imports`] (not re-resolved per import); `None` when the image has no
/// `DT_VERSYM`. Each per-symbol read is still individually bounds-checked by the
/// view.
fn version_of(
    view: &dyn ElfView,
    versions: &VersionMap,
    versym_cursor: Option<u64>,
    sym_index: u32,
) -> Result<(Option<Arc<str>>, Arc<str>)> {
    let empty: Arc<str> = Arc::from("");
    let Some(versym_cursor) = versym_cursor else {
        return Ok((None, empty));
    };
    let raw = read_u16(view, versym_cursor + u64::from(sym_index) * 2)?;
    let index = raw & VERSYM_VERSION_MASK;
    if index <= VER_NDX_GLOBAL {
        return Ok((None, empty));
    }
    match versions.get(&index) {
        Some((version, soname)) => Ok((Some(version.clone()), soname.clone())),
        None => Ok((None, empty)),
    }
}

/// Walk one `Elf64_Rela` table, emitting an import for every entry whose type is
/// `want_type` (a jump-slot or glob-dat relocation).
#[allow(clippy::too_many_arguments)]
fn walk_rela(
    view: &dyn ElfView,
    versym_cursor: Option<u64>,
    table: u64,
    table_size: u64,
    entsize: u64,
    want_type: u32,
    symtab: u64,
    strtab: u64,
    strsz: u64,
    versions: &VersionMap,
    out: &mut Vec<RawImport>,
) -> Result<()> {
    if entsize == 0 {
        return Err(Error::Malformed("zero relocation entry size"));
    }
    let count = table_size / entsize;
    // Fail loudly on an implausibly large table instead of truncating the walk:
    // a silent cap would drop the tail of `.rela.dyn`, where `-z combreloc` places
    // the symbolic `GLOB_DAT` relocations, leaving those imports unhookable.
    if count > MAX_RELOCS {
        return Err(Error::Malformed("relocation table exceeds sanity cap"));
    }
    for i in 0..count {
        let entry = table + i * entsize;
        // Elf64_Rela: r_offset(0), r_info(8), r_addend(16).
        let r_offset = read_u64(view, entry)?;
        let r_info = read_u64(view, entry + 8)?;
        // Low 32 bits are the relocation type (masked, so the narrowing is exact).
        let r_type = u32::try_from(r_info & 0xffff_ffff).unwrap_or(u32::MAX);
        if r_type != want_type {
            continue;
        }
        let sym_index = u32::try_from(r_info >> 32)
            .map_err(|_| Error::Malformed("relocation symbol index overflow"))?;
        // Elf64_Sym: st_name(0), st_info(4), st_other(5), st_shndx(6), st_value(8).
        let sym_cursor = symtab + u64::from(sym_index) * SYM_SIZE;
        let st_name = read_u32(view, sym_cursor)?;
        if st_name == 0 {
            continue; // Unnamed relocation target (e.g. IRELATIVE-style); skip.
        }
        let name = read_str(view, strtab, strsz, st_name)?;
        let (version, library) = version_of(view, versions, versym_cursor, sym_index)?;
        out.push(RawImport {
            library,
            symbol: Some(Symbol::Name(name)),
            version,
            slot: usize::try_from(view.slot_address(r_offset)?)
                .map_err(|_| Error::Malformed("slot address overflow"))?,
            kind: ImportKind::Standard,
            authenticated: false,
        });
    }
    Ok(())
}

/// Enumerate every rebindable import slot of an ELF image behind `view`.
///
/// Walks `DT_JMPREL` (`.rela.plt`, `R_*_JUMP_SLOT` — function imports) and
/// `DT_RELA` (`.rela.dyn`, `R_*_GLOB_DAT` — data/function-address imports), exactly
/// like the reference `plthook_elf`. Each import is named via
/// `dynsym[R_SYM].st_name → dynstr` and carries its observed symbol version
pub(crate) fn parse_imports(view: &dyn ElfView) -> Result<Vec<RawImport>> {
    // Reject an unsupported machine up front (also validated at header time).
    if arch::elf_reloc_kinds(view.machine()).is_none() {
        return Err(Error::Unsupported(
            "ELF machine is not EM_X86_64 or EM_AARCH64",
        ));
    }
    let kinds = arch::elf_reloc_kinds(view.machine()).ok_or(Error::Unsupported(
        "ELF machine is not EM_X86_64 or EM_AARCH64",
    ))?;

    let dynamic = read_dynamic(view)?;
    if let Some(syment) = dynamic.syment
        && syment != SYM_SIZE
    {
        return Err(Error::Malformed("DT_SYMENT != sizeof(Elf64_Sym)"));
    }
    let symtab_ptr = dynamic
        .symtab
        .ok_or(Error::Malformed("missing DT_SYMTAB"))?;
    let strtab_ptr = dynamic
        .strtab
        .ok_or(Error::Malformed("missing DT_STRTAB"))?;
    let strsz = dynamic.strsz.ok_or(Error::Malformed("missing DT_STRSZ"))?;
    let symtab = view.dynptr_cursor(symtab_ptr)?;
    let strtab = view.dynptr_cursor(strtab_ptr)?;

    let versions = build_version_map(view, &dynamic, strtab, strsz)?;

    // Resolve and validate the `DT_VERSYM` table base once for the whole
    // enumeration (it is re-used per import); `None` when the image has no
    // `DT_VERSYM`. `DT_VERSYM` is relocated in place, so it uses `dynptr_cursor`.
    let versym_cursor = dynamic
        .versym
        .map(|ptr| view.dynptr_cursor(ptr))
        .transpose()?;

    let mut out = Vec::new();

    // `.rela.plt` — the PLT jump slots (function imports). Only the RELA form is
    // used on x86-64 / `AArch64`; a REL PLT is not something these arches emit.
    if let (Some(jmprel), Some(pltrelsz)) = (dynamic.jmprel, dynamic.pltrelsz)
        && dynamic.pltrel == Some(DT_PLTREL_IS_RELA)
    {
        walk_rela(
            view,
            versym_cursor,
            view.dynptr_cursor(jmprel)?,
            pltrelsz,
            RELA_SIZE,
            kinds.jump_slot,
            symtab,
            strtab,
            strsz,
            &versions,
            &mut out,
        )?;
    }

    // `.rela.dyn` — GOT data slots (`GLOB_DAT`), matching the reference plthook's
    // optional GLOB_DAT handling so a function imported *as data* is rebindable too.
    if let Some(rela) = dynamic.rela {
        let relasz = dynamic
            .relasz
            .ok_or(Error::Malformed("DT_RELA without DT_RELASZ"))?;
        let relaent = dynamic.relaent.unwrap_or(RELA_SIZE);
        walk_rela(
            view,
            versym_cursor,
            view.dynptr_cursor(rela)?,
            relasz,
            relaent,
            kinds.glob_dat,
            symtab,
            strtab,
            strsz,
            &versions,
            &mut out,
        )?;
    }

    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::panic,
    clippy::too_many_lines
)]
mod tests {
    use super::*;

    /// An offline [`ElfView`] over borrowed file bytes: dynamic pointers are
    /// link-time virtual addresses, translated to file offsets through `PT_LOAD`.
    /// This is the host-independent backing that lets the parser be tested on this
    /// Windows host against the real Blackmagic `.so` files (and against the
    /// always-present synthetic image built below). It borrows its bytes so the
    /// fuzz can corrupt a single owned buffer in place across many iterations
    /// without re-allocating a copy each time.
    struct FileView<'a> {
        bytes: &'a [u8],
        loads: Vec<Segment>,
        machine: u16,
        dynamic_off: u64,
    }

    /// Translate a link-time virtual address to a file offset through the loadable
    /// segments — the offline addressing mode used only by the file-bytes view.
    fn vaddr_to_offset(loads: &[Segment], vaddr: u64) -> Result<u64> {
        for seg in loads {
            if vaddr >= seg.vaddr && vaddr < seg.vaddr.saturating_add(seg.filesz) {
                return Ok(seg.offset + (vaddr - seg.vaddr));
            }
        }
        Err(Error::Malformed(
            "virtual address not covered by any PT_LOAD",
        ))
    }

    impl<'a> FileView<'a> {
        fn new(bytes: &'a [u8]) -> Result<Self> {
            let machine = validate_ehdr(bytes.get(..64).ok_or(Error::Malformed("too small"))?)?;
            // e_phoff(32), e_phentsize(54), e_phnum(56).
            let e_phoff = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
            let e_phentsize = u16::from_le_bytes(bytes[54..56].try_into().unwrap()) as usize;
            let e_phnum = u16::from_le_bytes(bytes[56..58].try_into().unwrap()) as usize;
            if e_phentsize != PHDR_SIZE {
                return Err(Error::Malformed("unexpected e_phentsize"));
            }
            let end = e_phoff
                .checked_add(
                    e_phnum
                        .checked_mul(PHDR_SIZE)
                        .ok_or(Error::Malformed("phnum overflow"))?,
                )
                .ok_or(Error::Malformed("phdr table overflow"))?;
            let table = bytes
                .get(e_phoff..end)
                .ok_or(Error::Malformed("phdr table out of range"))?;
            let ph = parse_program_headers(table, e_phnum)?;
            let dynamic_vaddr = ph.dynamic_vaddr.ok_or(Error::Malformed("no PT_DYNAMIC"))?;
            let dynamic_off = vaddr_to_offset(&ph.loads, dynamic_vaddr)?;
            Ok(Self {
                bytes,
                loads: ph.loads,
                machine,
                dynamic_off,
            })
        }
    }

    impl ElfView for FileView<'_> {
        fn machine(&self) -> u16 {
            self.machine
        }
        fn dynamic_cursor(&self) -> u64 {
            self.dynamic_off
        }
        fn dynptr_cursor(&self, d_ptr: u64) -> Result<u64> {
            vaddr_to_offset(&self.loads, d_ptr)
        }
        fn vaddr_cursor(&self, vaddr: u64) -> Result<u64> {
            // On-disk, every dynamic pointer — relocated-in-place or not — is a
            // link-time vaddr, so `DT_VERNEED` resolves exactly like the others.
            vaddr_to_offset(&self.loads, vaddr)
        }
        fn read(&self, cursor: u64, buf: &mut [u8]) -> Result<()> {
            let start = usize::try_from(cursor).map_err(|_| Error::Malformed("cursor overflow"))?;
            let end = start
                .checked_add(buf.len())
                .ok_or(Error::Malformed("read overflow"))?;
            let src = self
                .bytes
                .get(start..end)
                .ok_or(Error::Malformed("read past file end"))?;
            buf.copy_from_slice(src);
            Ok(())
        }
        fn slot_address(&self, r_offset: u64) -> Result<u64> {
            // Offline: the slot's link-time virtual address (never patched here).
            Ok(r_offset)
        }
    }

    /// A hand-built, fully-valid ELF64 image for deterministic, **always-present**
    /// parser coverage that needs no external SDK — so the parser (including the
    /// whole version path and both relocation tables) has real, non-skipping
    /// coverage on every host and on a clean CI. It plants one `.rela.plt`
    /// `JUMP_SLOT` function import and one `.rela.dyn` `GLOB_DAT` data import, each
    /// versioned through `DT_VERSYM`/`DT_VERNEED`, and deliberately maps at a load
    /// base where vaddr != file offset so the offline `vaddr→offset` translation
    /// (and `vaddr_cursor` for `DT_VERNEED`) is exercised too.
    mod synthetic {
        // Imported directly from the defining `elf` module (this module's
        // grandparent); explicit names keep clippy's `wildcard_imports` quiet.
        use super::super::{
            DT_JMPREL, DT_NULL, DT_PLTREL, DT_PLTRELSZ, DT_RELA, DT_RELAENT, DT_RELASZ, DT_STRSZ,
            DT_STRTAB, DT_SYMENT, DT_SYMTAB, DT_VERNEED, DT_VERNEEDNUM, DT_VERSYM, EI_CLASS,
            EI_DATA, ELFCLASS64, ELFDATA2LSB, ET_DYN, PHDR_SIZE, PT_DYNAMIC, PT_LOAD,
        };

        /// x86-64 GOT-data relocation type (`R_X86_64_GLOB_DAT`).
        const R_X86_64_GLOB_DAT: u32 = 6;
        /// x86-64 PLT jump-slot relocation type (`R_X86_64_JUMP_SLOT`).
        const R_X86_64_JUMP_SLOT: u32 = 7;
        /// Load base: nonzero and distinct from any file offset, so `vaddr != off`.
        const VADDR_BASE: u64 = 0x20_0000;

        /// The built image plus the two import slot addresses it plants, so the
        /// test can assert `slot_address` returned exactly them.
        pub(super) struct Synthetic {
            pub(super) bytes: Vec<u8>,
            pub(super) fn_slot: u64,
            pub(super) data_slot: u64,
        }

        /// Little-endian byte assembler tracking the current file offset.
        struct Buf {
            b: Vec<u8>,
        }
        impl Buf {
            fn new() -> Self {
                Self { b: Vec::new() }
            }
            fn len(&self) -> usize {
                self.b.len()
            }
            fn zeros(&mut self, n: usize) {
                self.b.resize(self.b.len() + n, 0);
            }
            fn align(&mut self, a: usize) {
                while !self.b.len().is_multiple_of(a) {
                    self.b.push(0);
                }
            }
            fn u8(&mut self, v: u8) {
                self.b.push(v);
            }
            fn u16(&mut self, v: u16) {
                self.b.extend_from_slice(&v.to_le_bytes());
            }
            fn u32(&mut self, v: u32) {
                self.b.extend_from_slice(&v.to_le_bytes());
            }
            fn u64(&mut self, v: u64) {
                self.b.extend_from_slice(&v.to_le_bytes());
            }
            fn i64(&mut self, v: i64) {
                self.b.extend_from_slice(&v.to_le_bytes());
            }
            fn dynentry(&mut self, tag: i64, val: u64) {
                self.i64(tag);
                self.u64(val);
            }
            fn put_u16(&mut self, at: usize, v: u16) {
                self.b[at..at + 2].copy_from_slice(&v.to_le_bytes());
            }
            fn put_u32(&mut self, at: usize, v: u32) {
                self.b[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
            fn put_u64(&mut self, at: usize, v: u64) {
                self.b[at..at + 8].copy_from_slice(&v.to_le_bytes());
            }
        }

        fn vaddr(off: usize) -> u64 {
            VADDR_BASE + off as u64
        }

        /// Append a NUL-terminated string to a string table, returning its offset.
        fn add_str(dynstr: &mut Vec<u8>, s: &str) -> u32 {
            let off = u32::try_from(dynstr.len()).unwrap();
            dynstr.extend_from_slice(s.as_bytes());
            dynstr.push(0);
            off
        }

        pub(super) fn build() -> Synthetic {
            // dynstr, built standalone (its offsets are relative to its own start).
            let mut dynstr: Vec<u8> = vec![0]; // index 0 must be the empty string.
            let name_fn = add_str(&mut dynstr, "hookfs_fn");
            let name_data = add_str(&mut dynstr, "hookfs_data");
            let ver_a = add_str(&mut dynstr, "GLIBC_2.2.5");
            let ver_b = add_str(&mut dynstr, "GLIBC_2.34");
            let soname = add_str(&mut dynstr, "libc.so.6");

            let mut s = Buf::new();
            // Reserve the ELF header (64 bytes) + 2 program headers; patched last.
            s.zeros(64 + 2 * PHDR_SIZE);

            // GOT: the two slots the relocations target (values irrelevant offline).
            s.align(8);
            let got_off = s.len();
            s.u64(0);
            s.u64(0);
            let fn_slot = vaddr(got_off);
            let data_slot = vaddr(got_off + 8);

            // .dynsym: sym[0] = null, sym[1] = hookfs_fn (fn), sym[2] = hookfs_data.
            // Elf64_Sym: st_name(0,u32) st_info(4) st_other(5) st_shndx(6,u16)
            //            st_value(8,u64) st_size(16,u64) = 24 bytes.
            s.align(8);
            let dynsym_off = s.len();
            s.zeros(24); // sym[0]
            s.u32(name_fn);
            s.u8(0x12); // STB_GLOBAL | STT_FUNC
            s.u8(0);
            s.u16(0); // SHN_UNDEF (imported)
            s.u64(0);
            s.u64(0);
            s.u32(name_data);
            s.u8(0x11); // STB_GLOBAL | STT_OBJECT
            s.u8(0);
            s.u16(0);
            s.u64(0);
            s.u64(0);

            // .dynstr.
            let dynstr_off = s.len();
            s.b.extend_from_slice(&dynstr);
            let dynstr_len = dynstr.len() as u64;

            // .rela.plt: one JUMP_SLOT for sym index 1 (hookfs_fn).
            // Elf64_Rela: r_offset(0,u64) r_info(8,u64) r_addend(16,i64) = 24 bytes.
            s.align(8);
            let relaplt_off = s.len();
            s.u64(fn_slot);
            s.u64((1u64 << 32) | u64::from(R_X86_64_JUMP_SLOT));
            s.i64(0);

            // .rela.dyn: one GLOB_DAT for sym index 2 (hookfs_data).
            s.align(8);
            let reladyn_off = s.len();
            s.u64(data_slot);
            s.u64((2u64 << 32) | u64::from(R_X86_64_GLOB_DAT));
            s.i64(0);

            // .gnu.version (versym): one u16 per dynsym entry.
            s.align(2);
            let versym_off = s.len();
            s.u16(0); // sym[0]
            s.u16(2); // sym[1] -> version index 2 (GLIBC_2.2.5)
            s.u16(3); // sym[2] -> version index 3 (GLIBC_2.34)

            // .gnu.version_r (verneed): one Verneed for libc.so.6 with two Vernaux.
            s.align(4);
            let verneed_off = s.len();
            // Elf64_Verneed: vn_version(0,u16) vn_cnt(2,u16) vn_file(4,u32)
            //                vn_aux(8,u32) vn_next(12,u32) = 16 bytes.
            s.u16(1);
            s.u16(2);
            s.u32(soname);
            s.u32(16); // vn_aux -> first Vernaux directly after the Verneed
            s.u32(0); // vn_next -> only one Verneed
            // Elf64_Vernaux: vna_hash(0,u32) vna_flags(4,u16) vna_other(6,u16)
            //                vna_name(8,u32) vna_next(12,u32) = 16 bytes.
            s.u32(0x0b0b_0b0b);
            s.u16(0);
            s.u16(2); // vna_other -> version index 2
            s.u32(ver_a);
            s.u32(16); // vna_next -> next Vernaux
            s.u32(0x0c0c_0c0c);
            s.u16(0);
            s.u16(3); // vna_other -> version index 3
            s.u32(ver_b);
            s.u32(0); // vna_next -> last

            // .dynamic.
            s.align(8);
            let dynamic_off = s.len();
            s.dynentry(DT_SYMTAB, vaddr(dynsym_off));
            s.dynentry(DT_SYMENT, 24);
            s.dynentry(DT_STRTAB, vaddr(dynstr_off));
            s.dynentry(DT_STRSZ, dynstr_len);
            s.dynentry(DT_JMPREL, vaddr(relaplt_off));
            s.dynentry(DT_PLTRELSZ, 24);
            s.dynentry(DT_PLTREL, DT_RELA as u64);
            s.dynentry(DT_RELA, vaddr(reladyn_off));
            s.dynentry(DT_RELASZ, 24);
            s.dynentry(DT_RELAENT, 24);
            s.dynentry(DT_VERSYM, vaddr(versym_off));
            s.dynentry(DT_VERNEED, vaddr(verneed_off));
            s.dynentry(DT_VERNEEDNUM, 1);
            s.dynentry(DT_NULL, 0);
            let dynamic_size = (s.len() - dynamic_off) as u64;

            let total = s.len() as u64;

            // ---- Patch the ELF header (offset 0). ----
            s.b[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
            s.b[EI_CLASS] = ELFCLASS64;
            s.b[EI_DATA] = ELFDATA2LSB;
            s.b[6] = 1; // EI_VERSION
            s.put_u16(16, ET_DYN);
            s.put_u16(18, crate::arch::EM_X86_64);
            s.put_u32(20, 1); // e_version
            s.put_u64(32, 64); // e_phoff
            s.put_u16(52, 64); // e_ehsize
            s.put_u16(54, PHDR_SIZE as u16); // e_phentsize
            s.put_u16(56, 2); // e_phnum

            // ---- Patch the program headers. ----
            // PT_LOAD covering the whole file at VADDR_BASE (offset 0 -> VADDR_BASE).
            let p0 = 64;
            s.put_u32(p0, PT_LOAD);
            s.put_u32(p0 + 4, 5); // R+X
            s.put_u64(p0 + 8, 0); // p_offset
            s.put_u64(p0 + 16, VADDR_BASE); // p_vaddr
            s.put_u64(p0 + 24, VADDR_BASE); // p_paddr
            s.put_u64(p0 + 32, total); // p_filesz
            s.put_u64(p0 + 40, total); // p_memsz
            s.put_u64(p0 + 48, 0x1000); // p_align
            // PT_DYNAMIC.
            let p1 = 64 + PHDR_SIZE;
            s.put_u32(p1, PT_DYNAMIC);
            s.put_u32(p1 + 4, 6); // R+W
            s.put_u64(p1 + 8, dynamic_off as u64); // p_offset
            s.put_u64(p1 + 16, vaddr(dynamic_off)); // p_vaddr
            s.put_u64(p1 + 24, vaddr(dynamic_off)); // p_paddr
            s.put_u64(p1 + 32, dynamic_size); // p_filesz
            s.put_u64(p1 + 40, dynamic_size); // p_memsz
            s.put_u64(p1 + 48, 8); // p_align

            Synthetic {
                bytes: s.b,
                fn_slot,
                data_slot,
            }
        }
    }

    /// Locate a real BRAW `.so` from the sibling `braw-rs` SDK checkout, resolved
    /// relative to this crate (`CARGO_MANIFEST_DIR/../../../braw-rs`) so it works on
    /// any host (Windows dev, WSL, native Linux) and honors a `PLTHOOK_BRAW_DIR`
    /// override. `None` only when the license-restricted checkout is genuinely
    /// absent (a clean CI) — where `parses_synthetic_elf_full_version_and_reloc_path`
    /// provides the guaranteed, non-skipping parser coverage instead.
    fn braw_so(name: &str) -> Option<Vec<u8>> {
        let dir = std::env::var_os("PLTHOOK_BRAW_DIR").map_or_else(
            || std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../braw-rs"),
            std::path::PathBuf::from,
        );
        std::fs::read(dir.join(name)).ok()
    }

    fn names(imports: &[RawImport]) -> std::collections::BTreeSet<String> {
        imports
            .iter()
            .filter_map(|i| i.symbol.as_ref())
            .map(Symbol::describe)
            .collect()
    }

    #[test]
    fn parses_synthetic_elf_full_version_and_reloc_path() {
        // Always-present, host-independent coverage: header, program headers, the
        // dynamic array, symtab, strtab, both relocation tables, `DT_VERSYM`, and
        // `DT_VERNEED` (via `vaddr_cursor`) in one parse — no external SDK required.
        let syn = synthetic::build();
        let view = FileView::new(&syn.bytes).expect("valid synthetic ELF64 image");
        assert_eq!(view.machine, arch::EM_X86_64);
        let imports = parse_imports(&view).expect("enumerate synthetic imports");

        // Exactly the two imports planted: one JUMP_SLOT fn, one GLOB_DAT data.
        let set = names(&imports);
        let expected: std::collections::BTreeSet<String> = ["hookfs_data", "hookfs_fn"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        assert_eq!(set, expected);

        let f = imports
            .iter()
            .find(|i| i.symbol == Some(Symbol::name("hookfs_fn")))
            .expect("hookfs_fn import (JUMP_SLOT)");
        assert_eq!(
            f.version.as_deref(),
            Some("GLIBC_2.2.5"),
            "hookfs_fn@GLIBC_2.2.5"
        );
        assert_eq!(&*f.library, "libc.so.6", "provider SONAME from DT_VERNEED");
        assert_eq!(f.slot as u64, syn.fn_slot, "JUMP_SLOT slot address");

        let d = imports
            .iter()
            .find(|i| i.symbol == Some(Symbol::name("hookfs_data")))
            .expect("hookfs_data import (GLOB_DAT)");
        assert_eq!(
            d.version.as_deref(),
            Some("GLIBC_2.34"),
            "hookfs_data@GLIBC_2.34"
        );
        assert_eq!(&*d.library, "libc.so.6");
        assert_eq!(d.slot as u64, syn.data_slot, "GLOB_DAT slot address");
    }

    /// Overwrite the value of the `.dynamic` entry `(tag, old)` with `new`. Matches
    /// the full 16-byte entry (tag + current value) so it targets the intended tag
    /// unambiguously — a tag-only scan would collide with e.g. a `p_align == 8`.
    fn patch_dyn_value(bytes: &mut [u8], tag: i64, old: u64, new: u64) {
        let mut pat = Vec::with_capacity(16);
        pat.extend_from_slice(&u64::try_from(tag).unwrap().to_le_bytes());
        pat.extend_from_slice(&old.to_le_bytes());
        for off in (0..=bytes.len().saturating_sub(16)).step_by(8) {
            if bytes[off..off + 16] == pat[..] {
                bytes[off + 8..off + 16].copy_from_slice(&new.to_le_bytes());
                return;
            }
        }
        panic!("dynamic entry (tag {tag}, val {old}) not found");
    }

    #[test]
    fn oversized_relocation_table_is_rejected_not_truncated() {
        // A corrupt/implausibly large `DT_RELASZ` must fail loudly rather than
        // silently truncate the `.rela.dyn` walk — a silent cap would drop the
        // `GLOB_DAT` relocations `-z combreloc` sorts to the table's tail, leaving
        // those imports unhookable with no diagnostic. `DT_RELASZ` is 24 in the
        // synthetic image; raise it to imply far more than `MAX_RELOCS` entries.
        let mut syn = synthetic::build();
        patch_dyn_value(&mut syn.bytes, DT_RELASZ, 24, 0xFFFF_FF00);
        let view = FileView::new(&syn.bytes).expect("header still valid");
        match parse_imports(&view) {
            Err(Error::Malformed(msg)) => {
                assert!(msg.contains("sanity cap"), "got: {msg}");
            }
            other => panic!("expected an oversized-table rejection, got {other:?}"),
        }
    }

    #[test]
    fn parses_main_lib_fs_imports_with_versions_and_reloc_types() {
        let Some(bytes) = braw_so("libBlackmagicRawAPI.so") else {
            eprintln!("skipping (real-world validation): libBlackmagicRawAPI.so not found");
            return;
        };
        let view = FileView::new(&bytes).expect("valid ELF64 image");
        assert_eq!(view.machine, arch::EM_X86_64);
        let imports = parse_imports(&view).expect("enumerate imports");
        let set = names(&imports);

        // audit manifest records it.
        for sym in [
            "fopen",
            "fclose",
            "fread",
            "fwrite",
            "fseek",
            "fseeko",
            "ftello",
            "fflush",
            "open",
            "open64",
            "close",
            "read",
            "readv",
            "writev",
            "lseek",
            "__xstat",
            "__fxstat",
            "fstatvfs",
            "fsync",
            "ftruncate",
            "ftruncate64",
            "realpath",
            "mkdir",
            "remove",
            "rename",
            "fprintf",
        ] {
            assert!(
                set.contains(sym),
                "expected import `{sym}` in libBlackmagicRawAPI.so"
            );
        }
        // Loader symbols are present too (hooked only for auto_rescan / passthrough).
        for sym in ["dlopen", "dlsym", "dlclose", "dladdr"] {
            assert!(set.contains(sym), "expected loader import `{sym}`");
        }

        // Every enumerated import is a JUMP_SLOT/GLOB_DAT-sourced entry with a name.
        assert!(imports.iter().all(|i| i.symbol.is_some()));

        // Symbol versions are retained (R2): the libc imports carry GLIBC_*, and
        // `realpath` specifically is GLIBC_2.3 per the manifest.
        let open = imports
            .iter()
            .find(|i| i.symbol == Some(Symbol::name("open")))
            .expect("open import");
        assert_eq!(
            open.version.as_deref(),
            Some("GLIBC_2.2.5"),
            "open@GLIBC_2.2.5"
        );
        assert!(
            open.library.contains("libc.so"),
            "provider SONAME resolved: {}",
            open.library
        );

        let realpath = imports
            .iter()
            .find(|i| i.symbol == Some(Symbol::name("realpath")))
            .expect("realpath import");
        assert_eq!(
            realpath.version.as_deref(),
            Some("GLIBC_2.3"),
            "realpath@GLIBC_2.3"
        );
    }

    #[test]
    fn parses_decoder_ftell_import() {
        let Some(bytes) = braw_so("libDecoderCUDA.so") else {
            eprintln!("skipping (real-world validation): libDecoderCUDA.so not found");
            return;
        };
        let view = FileView::new(&bytes).expect("valid ELF64");
        let set = names(&parse_imports(&view).expect("imports"));
        assert!(set.contains("ftell"), "libDecoderCUDA.so imports ftell");
        assert!(set.contains("fopen"));
        assert!(set.contains("fread"));
    }

    #[test]
    fn parses_libcxx_dir_imports() {
        let Some(bytes) = braw_so("libc++.so.1") else {
            eprintln!("skipping (real-world validation): libc++.so.1 not found");
            return;
        };
        let view = FileView::new(&bytes).expect("valid ELF64");
        let set = names(&parse_imports(&view).expect("imports"));
        // libc++ is where C++ streams bottom out: it imports the dir family.
        for sym in ["opendir", "readdir", "closedir", "__xstat"] {
            assert!(set.contains(sym), "libc++.so.1 imports `{sym}`");
        }
    }

    #[test]
    fn malformed_images_never_read_out_of_bounds() {
        // A small deterministic xorshift PRNG keeps the fuzz dependency-free.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // Wholly random buffers must never panic: every outcome is a typed Result.
        for _ in 0..8000 {
            let len = (next() as usize % 4096) + 1;
            let mut buf = vec![0u8; len];
            for b in &mut buf {
                *b = next() as u8;
            }
            let _ = FileView::new(&buf); // may Err; must not panic or read OOB.
        }

        // Corrupt a valid image across its **whole length** — not just the first
        // few KiB — so the deep symtab / strtab / rela / versym / verneed regions
        // are fuzzed too, proving the structural bounds checks keep every read in
        // range no matter which table a flip lands in. Corruption is done in place
        // on one owned buffer (restored each iteration) to avoid re-allocating.
        //
        // A worked-through corruption of one such deep region is exactly what
        // surfaced the `DT_VERNEED` addressing asymmetry on the live path.
        let mut fuzz_whole = |base: &mut Vec<u8>, singles: usize, bursts: usize| {
            let len = base.len();
            // Single-byte flips anywhere in the image.
            for _ in 0..singles {
                let idx = next() as usize % len;
                let orig = base[idx];
                base[idx] ^= (next() as u8) | 1; // guarantee an actual change
                if let Ok(view) = FileView::new(base) {
                    let _ = parse_imports(&view); // Ok or Err — never a panic/OOB.
                }
                base[idx] = orig;
            }
            // Multi-byte bursts (correlated corruption across several fields).
            for _ in 0..bursts {
                let flips = (next() as usize % 24) + 1;
                let mut saved: Vec<(usize, u8)> = Vec::with_capacity(flips);
                for _ in 0..flips {
                    let idx = next() as usize % len;
                    saved.push((idx, base[idx]));
                    base[idx] ^= next() as u8;
                }
                if let Ok(view) = FileView::new(base) {
                    let _ = parse_imports(&view);
                }
                // Restore in reverse so repeated indices unwind to the original.
                for (idx, v) in saved.into_iter().rev() {
                    base[idx] = v;
                }
            }
        };

        // The synthetic image is small and dense — every flip lands in a real
        // header/table, so this heavily exercises deep-table corruption fast and
        // is always available (no external SDK needed).
        let mut synth = synthetic::build().bytes;
        fuzz_whole(&mut synth, 40_000, 20_000);

        // A large, realistic image when the SDK is present (bigger tables, more
        // string/version entries) — moderate iterations, whole-image coverage.
        if let Some(mut base) = braw_so("libBlackmagicRawAPI.so") {
            fuzz_whole(&mut base, 12_000, 4_000);
        }
    }
}
