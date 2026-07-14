//!
//! This module decodes the dynamic-import metadata of a 64-bit Mach-O image and
//! yields one [`RawImport`] per rebindable symbol-pointer / fixup slot. It uses no
//! **every** offset, count, string, and slot address is validated against the
//! image's file ranges *before* the byte is read, so a malformed image yields a
//! typed [`Error`] and never a read out of bounds.
//!
//! Imports are recovered from up to three sources and merged (deduplicated by slot
//! address), so the inventory is complete however a slice was linked:
//!
//! 1. **Indirect symbol table (fishhook-style), PRIMARY.** Each
//!    `S_LAZY_SYMBOL_POINTERS` / `S_NON_LAZY_SYMBOL_POINTERS` section maps its
//!    pointer slots through `reserved1` into the indirect symbol table
//!    (`LC_DYSYMTAB.indirectsymoff`) → `LC_SYMTAB` nlist → string table name. This
//!    is pure metadata (all in `__LINKEDIT` / the section headers), so it works
//!    entirely from the live in-memory image — the reason fishhook keeps working on
//!    modern chained-fixups binaries.
//! 2. **Chained fixups (`LC_DYLD_CHAINED_FIXUPS`, macOS 12+/iOS 15+), COMPLEMENT.**
//!    The `dyld_chained_fixups_header` → imports array + symbol pool + per-page
//!    fixup chains, per Apple `<mach-o/fixup-chains.h>`. Supports
//!    `DYLD_CHAINED_PTR_64` / `_64_OFFSET` and the arm64e formats. The chain
//!    *metadata* lives in `__LINKEDIT` (intact in memory), but the per-slot chain
//!    encodings live in `__DATA*` and are **overwritten** by dyld once fixups are
//!    applied — so on a live image the chain walk reads the original encodings from
//!    the on-disk file ([`MachOView::read_slot_encoding`]); on synthetic/offline
//!    bytes both come from the same buffer.
//! 3. **Legacy bind opcodes (`LC_DYLD_INFO[_ONLY]`), COMPLEMENT.** The
//!    bind/weak-bind/lazy-bind ULEB/SLEB opcode streams (`<mach-o/loader.h>`). These
//!    encode the slot `(segment, offset)` directly, so — unlike chained fixups —
//!    they need no data-segment read and work from memory.
//!
//! Recovered names are normalized by [`normalize_symbol_name`]: a leading `_` is
//! stripped (every Mach-O C symbol carries it) and a trailing `$`-variant suffix is
//! dropped (`_stat$INODE64` → `stat`), so a request for `stat` matches the x86-64
//! `$INODE64` alias and the arm64 bare name uniformly.
//!
//! # arm64e (R1)
//! Authenticated (PAC-signed) bind slots are detected — via the arm64e chain
//! pointer formats and their per-slot `auth` bit — and flagged
//! [`RawImport::authenticated`]. The transaction ([`crate::slot`]) then refuses to
//! rebind them rather than write a pointer that would fail its `AUT*` check
//!
//! # Host-independent by construction
//! Like the ELF parser, all format logic runs through the [`MachOView`] trait, so
//! the same code serves both the live macOS image ([`crate::module`]) and the
//! synthetic byte fixtures in the tests below — the latter exercising every path on
//! this Windows host.

use crate::arch;
use crate::error::{Error, Result};
use crate::import::RawImport;
use crate::{ImportKind, Symbol};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// ---- Mach-O constants ------------------------------------------------------

/// 64-bit Mach-O header magic (host-endian little: `MH_MAGIC_64`).
pub(crate) const MH_MAGIC_64: u32 = 0xfeed_facf;
/// 32-bit Mach-O header magic (`MH_MAGIC`) — rejected (we only rebind 64-bit).
const MH_MAGIC_32: u32 = 0xfeed_face;
/// Byte-swapped 64-/32-bit magics (`MH_CIGAM_64` / `MH_CIGAM`) — a foreign-endian
/// image, rejected (the SDK ships only little-endian slices).
const MH_CIGAM_64: u32 = 0xcffa_edfe;
const MH_CIGAM_32: u32 = 0xcefa_edfe;

/// Universal ("fat") header magic, big-endian (`FAT_MAGIC` / `FAT_MAGIC_64`).
pub(crate) const FAT_MAGIC: u32 = 0xcafe_babe;
pub(crate) const FAT_MAGIC_64: u32 = 0xcafe_babf;

/// Size of `mach_header_64`.
const MACH_HEADER_64_SIZE: u64 = 32;
/// Size of `segment_command_64`'s fixed head (sections follow it).
const SEGMENT_COMMAND_64_SIZE: u64 = 72;
/// Size of one `section_64`.
const SECTION_64_SIZE: u64 = 80;
/// Size of one `nlist_64`.
const NLIST_64_SIZE: u64 = 16;

// Load-command ids (`<mach-o/loader.h>`). `LC_REQ_DYLD` (0x8000_0000) is OR-ed
// into the commands the dynamic linker must understand.
const LC_REQ_DYLD: u32 = 0x8000_0000;
const LC_SEGMENT_64: u32 = 0x19;
const LC_SYMTAB: u32 = 0x2;
const LC_DYSYMTAB: u32 = 0xb;
const LC_LOAD_DYLIB: u32 = 0xc;
const LC_LOAD_WEAK_DYLIB: u32 = 0x18 | LC_REQ_DYLD;
const LC_REEXPORT_DYLIB: u32 = 0x1f | LC_REQ_DYLD;
const LC_LOAD_UPWARD_DYLIB: u32 = 0x23 | LC_REQ_DYLD;
const LC_LAZY_LOAD_DYLIB: u32 = 0x20;
const LC_DYLD_INFO: u32 = 0x22;
const LC_DYLD_INFO_ONLY: u32 = 0x22 | LC_REQ_DYLD;
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x34 | LC_REQ_DYLD;

/// Section-type mask (`SECTION_TYPE`) and the two symbol-pointer section types.
const SECTION_TYPE: u32 = 0xff;
const S_NON_LAZY_SYMBOL_POINTERS: u32 = 0x6;
const S_LAZY_SYMBOL_POINTERS: u32 = 0x7;

/// Indirect-symbol-table sentinel indices (`<mach-o/loader.h>`): a slot that binds
/// to a local or absolute symbol carries no import name.
const INDIRECT_SYMBOL_LOCAL: u32 = 0x8000_0000;
const INDIRECT_SYMBOL_ABS: u32 = 0x4000_0000;

// nlist flags.
const N_STAB: u8 = 0xe0;
const N_TYPE: u8 = 0x0e;
const N_UNDF: u8 = 0x00;
const N_EXT: u8 = 0x01;

// Two-level-namespace / bind special library ordinals.
const BIND_SPECIAL_DYLIB_SELF: i64 = 0;

// Bind opcodes (`<mach-o/loader.h>`).
const BIND_OPCODE_MASK: u8 = 0xf0;
const BIND_IMMEDIATE_MASK: u8 = 0x0f;
const BIND_OPCODE_DONE: u8 = 0x00;
const BIND_OPCODE_SET_DYLIB_ORDINAL_IMM: u8 = 0x10;
const BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB: u8 = 0x20;
const BIND_OPCODE_SET_DYLIB_SPECIAL_IMM: u8 = 0x30;
const BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM: u8 = 0x40;
const BIND_OPCODE_SET_TYPE_IMM: u8 = 0x50;
const BIND_OPCODE_SET_ADDEND_SLEB: u8 = 0x60;
const BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB: u8 = 0x70;
const BIND_OPCODE_ADD_ADDR_ULEB: u8 = 0x80;
const BIND_OPCODE_DO_BIND: u8 = 0x90;
const BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB: u8 = 0xa0;
const BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED: u8 = 0xb0;
const BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB: u8 = 0xc0;

// Chained-fixups constants (`<mach-o/fixup-chains.h>`).
const DYLD_CHAINED_PTR_START_NONE: u16 = 0xffff;
const DYLD_CHAINED_PTR_ARM64E: u16 = 1;
const DYLD_CHAINED_PTR_64: u16 = 2;
const DYLD_CHAINED_PTR_64_OFFSET: u16 = 6;
const DYLD_CHAINED_PTR_ARM64E_KERNEL: u16 = 7;
const DYLD_CHAINED_PTR_ARM64E_USERLAND: u16 = 9;
const DYLD_CHAINED_PTR_ARM64E_FIRMWARE: u16 = 10;
const DYLD_CHAINED_PTR_ARM64E_USERLAND24: u16 = 12;
const DYLD_CHAINED_IMPORT: u32 = 1;
const DYLD_CHAINED_IMPORT_ADDEND: u32 = 2;
const DYLD_CHAINED_IMPORT_ADDEND64: u32 = 3;

/// Defensive iteration bounds — a conformant image stays far below these; exceeding
/// one means the tables are malformed (or maliciously unterminated), so the parser
/// stops with [`Error::Malformed`] rather than looping.
const MAX_COMMANDS: u32 = 1 << 16;
const MAX_SECTIONS: u32 = 1 << 16;
const MAX_INDIRECT: u32 = 1 << 24;
const MAX_BIND_BYTES: u64 = 1 << 24;
const MAX_CHAIN_STEPS: u64 = 1 << 24;
const MAX_FIXUP_SEGMENTS: u32 = 1 << 12;
const MAX_NAME_LEN: usize = 8192;
const MAX_DYLIBS: usize = 1 << 12;
const ULEB_MAX_SHIFT: u32 = 63;

// ---- The addressing-mode abstraction ---------------------------------------

/// One `LC_SEGMENT_64` of a Mach-O image, enough to translate a file offset to a
/// live patch address and to bound-check a slot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Segment {
    /// Link-time virtual address of the segment start (`vmaddr`).
    pub(crate) vmaddr: u64,
    /// In-memory size (`vmsize`) — the bound for a live read; consulted only by the
    /// live Darwin engine ([`crate::module`], macOS + iOS).
    #[cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(dead_code))]
    pub(crate) vmsize: u64,
    /// File offset of the segment start (`fileoff`).
    pub(crate) fileoff: u64,
    /// On-disk size (`filesize`).
    pub(crate) filesize: u64,
}

impl Segment {
    /// Whether file offset `off` falls inside this segment's on-disk range. Used by
    /// the offline (test) view's `slot_address`; the live engine inlines the same
    /// check in its own file-offset → address translation.
    #[cfg(test)]
    fn contains_fileoff(&self, off: u64) -> bool {
        off >= self.fileoff && off < self.fileoff.saturating_add(self.filesize)
    }
}

/// The addressing-mode abstraction over a **thin** 64-bit Mach-O slice, so the same
/// format logic serves both a live in-memory image and offline file bytes.
///
/// A *cursor* is always a **file offset** (relative to the start of the thin
/// slice). The implementation decides how to fetch bytes at that offset:
/// * offline (synthetic tests): read directly from the byte buffer;
/// * live (the macOS runtime): translate the file offset to an absolute address
///   through the containing segment (`vmaddr - fileoff + slide + off`) and read
///   process memory.
pub(crate) trait MachOView {
    /// The slice's `cputype` (picks the supported-architecture check and, with the
    /// chain pointer format, the arm64e authentication handling).
    fn cpu_type(&self) -> u32;

    /// The slice's `cpusubtype` (its capability bits mask off; the low bits give
    /// arm64e).
    fn cpu_subtype(&self) -> u32;

    /// Whether this is an Apple **arm64e** slice (`CPU_TYPE_ARM64` +
    /// `CPU_SUBTYPE_ARM64E`). On arm64e the function-pointer GOT (`__auth_got`)
    /// holds authenticated (PAC-signed) pointers, so a slot reached through the
    /// indirect-symbol path — which cannot see a per-slot `auth` bit — must be
    /// fixup walk detects authentication precisely per slot instead.
    fn is_arm64e(&self) -> bool {
        self.cpu_type() == arch::CPU_TYPE_ARM64
            && (self.cpu_subtype() & !arch::CPU_SUBTYPE_MASK) == arch::CPU_SUBTYPE_ARM64E
    }

    /// Read exactly `buf.len()` bytes of image metadata at file offset `off`,
    /// bounds-checked against the slice. Serves headers, load commands, and
    /// `__LINKEDIT` (symbol/string/indirect/chained-fixups metadata) — all
    /// identical on disk and in memory. Fails ([`Error::Malformed`]) for any read
    /// that would leave the slice.
    fn read(&self, off: u64, buf: &mut [u8]) -> Result<()>;

    /// Whether the **original on-disk file bytes** are available for
    /// [`Self::read_slot_encoding`]. On a live image without a backing file (e.g. a
    /// dyld-shared-cache library) this is `false`, so the chained-fixups walk is
    /// skipped and the indirect-symbol path carries the enumeration.
    fn file_backed(&self) -> bool;

    /// Read the **original** (un-fixed-up) 8-byte chain encoding at data-segment
    /// file offset `off`. On a live image this reads the on-disk file (the
    /// in-memory slot has been overwritten by dyld); offline it reads the same
    /// buffer as [`Self::read`]. Only called when [`Self::file_backed`] is `true`.
    fn read_slot_encoding(&self, off: u64) -> Result<u64>;

    /// The live patch-target address of a slot at file offset `off` (for a live
    /// image `slide + vmaddr + (off - fileoff)` of the containing segment; offline
    /// the slot's link-time vmaddr). Validated to be a pointer-aligned cell inside a
    /// segment's mapped range.
    fn slot_address(&self, off: u64) -> Result<u64>;
}

// ---- Little-endian primitive reads through a view --------------------------

fn read_u16(view: &dyn MachOView, off: u64) -> Result<u16> {
    let mut b = [0u8; 2];
    view.read(off, &mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(view: &dyn MachOView, off: u64) -> Result<u32> {
    let mut b = [0u8; 4];
    view.read(off, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(view: &dyn MachOView, off: u64) -> Result<u64> {
    let mut b = [0u8; 8];
    view.read(off, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Read a NUL-terminated string starting at file offset `off`, capped at
/// [`MAX_NAME_LEN`]; every byte is range-checked by the view. Returns the decoded
/// (UTF-8-lossy) name and the **raw** number of name bytes consumed, *excluding* the
/// terminating NUL.
///
/// A caller walking an opcode/record stream MUST advance its cursor by this raw byte
/// count (+1 for the NUL), never by the returned `String`'s length:
/// [`String::from_utf8_lossy`] substitutes a 3-byte `U+FFFD` for each invalid byte,
/// so the decoded length can exceed the bytes actually read and would desync the
fn read_cstr_raw(view: &dyn MachOView, off: u64) -> Result<(String, u64)> {
    let mut bytes = Vec::new();
    for i in 0..=MAX_NAME_LEN as u64 {
        let mut byte = [0u8; 1];
        view.read(off + i, &mut byte)?;
        if byte[0] == 0 {
            // `i` is the index of the NUL, i.e. exactly the number of name bytes read.
            return Ok((String::from_utf8_lossy(&bytes).into_owned(), i));
        }
        bytes.push(byte[0]);
    }
    Err(Error::Malformed("unterminated Mach-O string"))
}

/// Read a NUL-terminated string starting at file offset `off`. Convenience wrapper
/// over [`read_cstr_raw`] for callers that only need the decoded name (not a cursor
/// advance).
fn read_cstr(view: &dyn MachOView, off: u64) -> Result<String> {
    Ok(read_cstr_raw(view, off)?.0)
}

/// Read a NUL-terminated name from a string table of known bounds: `base` is the
/// table's file offset, `size` its byte length, `strx` the entry's offset within
/// it. The offset is validated against `size` before any byte is read.
fn read_str_in_table(view: &dyn MachOView, base: u64, size: u64, strx: u32) -> Result<String> {
    let strx = u64::from(strx);
    if strx >= size {
        return Err(Error::Malformed("string-table offset past table size"));
    }
    let max = (size - strx).min(MAX_NAME_LEN as u64 + 1);
    let mut bytes = Vec::new();
    for i in 0..max {
        let mut byte = [0u8; 1];
        view.read(base + strx + i, &mut byte)?;
        if byte[0] == 0 {
            return Ok(String::from_utf8_lossy(&bytes).into_owned());
        }
        bytes.push(byte[0]);
    }
    Err(Error::Malformed("unterminated string-table entry"))
}

/// Normalize a Mach-O import symbol name to its match key (fishhook /
/// `plthook_osx`): strip a single leading `_`, then drop a trailing `$`-variant
/// suffix (`_stat$INODE64` → `stat`). Returns `None` for an empty result.
///
/// This makes a request for `stat` match the x86-64 `$INODE64` alias and the arm64
/// bare name identically, and lets the shared exact-match [`Symbol`] machinery in
/// [`crate::slot`] work unchanged across all three formats.
pub(crate) fn normalize_symbol_name(raw: &str) -> Option<String> {
    let without_underscore = raw.strip_prefix('_').unwrap_or(raw);
    let base = without_underscore
        .split('$')
        .next()
        .unwrap_or(without_underscore);
    if base.is_empty() {
        None
    } else {
        Some(base.to_owned())
    }
}

/// dyld's thread-local-storage bootstrap symbol, in [`normalize_symbol_name`] form.
///
/// Every thread-local's first descriptor word binds to the raw Mach-O symbol
/// `__tlv_bootstrap` (two leading underscores), which normalizes to
/// `_tlv_bootstrap` — and *only* that raw symbol normalizes to it, so matching the
/// normalized name reliably identifies it. Rebinding this slot would corrupt TLS
/// setup, so — exactly as `plthook_osx` hides the raw `__tlv_bootstrap` during
/// enumeration — no backend surfaces it as a rebindable import.
const TLV_BOOTSTRAP: &str = "_tlv_bootstrap";

// ---- Universal ("fat") container + thin-slice selection --------------------
//
// A loaded image's IN-MEMORY header is always a thin `mach_header_64` — dyld maps a
// single slice — but its ORIGINAL on-disk file, read by the live Darwin engine
// ([`crate::module`]) for the chained-fixup chain encodings, may be a UNIVERSAL
// ("fat") container holding several architecture slices at nonzero `fat_arch.offset`.
// [`classify`] decodes the container kind; [`thin_slice_range`] narrows a file to the
// loaded architecture's slice so a chain's thin-slice-relative file offsets index the
// correct bytes. Both are pure and host-independent, so ONE implementation serves the
// live macOS + iOS engine and the offline fat fixtures in the tests below — a single,

/// The byte offset and length of a thin slice inside a file.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliceRange {
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// One architecture slice of a universal container: its `(cpu_type, cpu_subtype)`
/// identity and byte range. `cpu_subtype` is retained because an arm64 slice and an
/// arm64e slice share `cpu_type` (`CPU_TYPE_ARM64`) and are told apart only by their
/// subtype — the live engine must select the exact loaded slice, or a chain's file
/// offsets would index the wrong bytes (the `plthook_osx` bug this avoids).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FatArch {
    pub(crate) cpu_type: u32,
    pub(crate) cpu_subtype: u32,
    pub(crate) range: SliceRange,
}

/// Detect the container kind of `bytes` (at least the first 4/8 bytes) and, for a
/// universal file, return the fat-arch table so a caller can pick a slice. A thin
/// 64-bit Mach-O reports `Thin`. Foreign-endian and 32-bit images are rejected.
#[derive(Debug)]
pub(crate) enum Container {
    /// A thin 64-bit Mach-O — parse it directly at offset 0.
    Thin,
    /// A universal binary — its slices, in file order.
    Fat(Vec<FatArch>),
}

/// Classify the image and, for a fat file, read the arch table from `bytes`.
pub(crate) fn classify(bytes: &[u8]) -> Result<Container> {
    let magic = read_slice_u32_le(bytes, 0)?;
    match magic {
        MH_MAGIC_64 => Ok(Container::Thin),
        MH_MAGIC_32 | MH_CIGAM_32 => Err(Error::Unsupported("32-bit Mach-O image")),
        MH_CIGAM_64 => Err(Error::Unsupported(
            "byte-swapped (foreign-endian) Mach-O image",
        )),
        _ => {
            // Fat headers are big-endian; both 32- and 64-bit fat arch tables exist.
            let be = read_slice_u32_be(bytes, 0)?;
            match be {
                FAT_MAGIC => parse_fat(bytes, false),
                FAT_MAGIC_64 => parse_fat(bytes, true),
                _ => Err(Error::Malformed("not a Mach-O or universal image")),
            }
        }
    }
}

/// Parse a universal header's arch table (all big-endian). `wide` selects the
/// 64-bit `fat_arch_64` (u64 offset/size) vs the 32-bit `fat_arch`. The
/// `cputype`/`cpusubtype` identity is captured alongside each slice's range so the
/// live engine can select the exact loaded arm64-vs-arm64e slice.
///
/// Every slice's `offset + size` is validated against the file up front (not just
/// the matched slice), so a corrupt sibling slice fails the whole parse rather than
/// all slices in-bounds; on failure the caller degrades to the in-memory
/// indirect-symbol path, so the stricter check never loses recoverable imports.
fn parse_fat(bytes: &[u8], wide: bool) -> Result<Container> {
    let nfat = read_slice_u32_be(bytes, 4)?;
    if nfat > 1024 {
        return Err(Error::Malformed("implausible fat arch count"));
    }
    let (entry_size, base) = if wide {
        (32usize, 8usize)
    } else {
        (20usize, 8usize)
    };
    let mut slices = Vec::new();
    for i in 0..nfat as usize {
        let entry = base + i * entry_size;
        // fat_arch{,_64}: cputype(0) cpusubtype(4) offset(8) size(...) align(...).
        let cpu_type = read_slice_u32_be(bytes, entry)?;
        let cpu_subtype = read_slice_u32_be(bytes, entry + 4)?;
        let (offset, size) = if wide {
            (
                read_slice_u64_be(bytes, entry + 8)?,
                read_slice_u64_be(bytes, entry + 16)?,
            )
        } else {
            (
                u64::from(read_slice_u32_be(bytes, entry + 8)?),
                u64::from(read_slice_u32_be(bytes, entry + 12)?),
            )
        };
        // Validate the slice lies inside the file before it is ever read.
        let end = offset
            .checked_add(size)
            .ok_or(Error::Malformed("fat slice overflow"))?;
        if end > bytes.len() as u64 {
            return Err(Error::Malformed("fat slice out of bounds"));
        }
        slices.push(FatArch {
            cpu_type,
            cpu_subtype,
            range: SliceRange { offset, size },
        });
    }
    Ok(Container::Fat(slices))
}

/// The byte range of the thin 64-bit Mach-O slice within `file` that matches the
/// loaded image's `(cpu_type, cpu_subtype)` — the shared fat-slice selector the live
/// engine ([`crate::module`]) uses to narrow a universal on-disk file before reading
/// the chained-fixup chain encodings.
///
/// * A **thin** file (`MH_MAGIC_64` at offset 0) is itself the slice → `0..len`.
/// * A **universal** file is searched (via [`classify`]) for the slice whose
///   `cpu_type` equals `cpu_type` and whose `cpu_subtype`, with the capability bits
///   masked off (matching [`MachOView::is_arm64e`]), equals `cpu_subtype` — i.e. the
///   exact slice dyld mapped, distinguishing an arm64 slice from an arm64e one. The
///   chosen slice's bounds were validated by [`parse_fat`]; it is additionally
///   confirmed to begin with a thin `MH_MAGIC_64` header.
///
/// Pure and host-independent (no live-image state), so it is exercised by the
/// synthetic fat fixtures in the unit tests below.
pub(crate) fn thin_slice_range(
    file: &[u8],
    cpu_type: u32,
    cpu_subtype: u32,
) -> Result<core::ops::Range<usize>> {
    match classify(file)? {
        Container::Thin => Ok(0..file.len()),
        Container::Fat(slices) => {
            let want_sub = cpu_subtype & !arch::CPU_SUBTYPE_MASK;
            for slice in slices {
                if slice.cpu_type != cpu_type
                    || (slice.cpu_subtype & !arch::CPU_SUBTYPE_MASK) != want_sub
                {
                    continue;
                }
                let start = usize::try_from(slice.range.offset)
                    .map_err(|_| Error::Malformed("fat slice offset overflow"))?;
                let end = slice
                    .range
                    .offset
                    .checked_add(slice.range.size)
                    .and_then(|e| usize::try_from(e).ok())
                    .ok_or(Error::Malformed("fat slice end overflow"))?;
                // `parse_fat` already validated `end <= file.len()`; confirm the
                // selected slice is a thin 64-bit Mach-O before handing back its range.
                if read_slice_u32_le(file, start)? != MH_MAGIC_64 {
                    return Err(Error::Malformed("fat slice is not a thin 64-bit Mach-O"));
                }
                return Ok(start..end);
            }
            Err(Error::Malformed(
                "no universal slice matches the loaded architecture",
            ))
        }
    }
}

// ---- Big-endian / little-endian reads over a borrowed byte slice -----------

fn read_slice_u32_le(bytes: &[u8], off: usize) -> Result<u32> {
    let arr: [u8; 4] = bytes
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::Malformed("read past slice bounds"))?;
    Ok(u32::from_le_bytes(arr))
}

fn read_slice_u32_be(bytes: &[u8], off: usize) -> Result<u32> {
    let arr: [u8; 4] = bytes
        .get(off..off + 4)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::Malformed("read past slice bounds"))?;
    Ok(u32::from_be_bytes(arr))
}

fn read_slice_u64_be(bytes: &[u8], off: usize) -> Result<u64> {
    let arr: [u8; 8] = bytes
        .get(off..off + 8)
        .and_then(|s| s.try_into().ok())
        .ok_or(Error::Malformed("read past slice bounds"))?;
    Ok(u64::from_be_bytes(arr))
}

// ---- Load-command walk -----------------------------------------------------

/// A `S_*_SYMBOL_POINTERS` section: enough to map its slots through the indirect
/// symbol table.
#[derive(Debug, Clone, Copy)]
struct PointerSection {
    /// File offset of the section's first pointer slot.
    fileoff: u64,
    /// Section byte size (`size`); slot count is `size / 8`.
    size: u64,
    /// `reserved1` — the section's base index into the indirect symbol table.
    indirect_index: u32,
}

/// The load-command metadata the import walk needs.
#[derive(Debug, Default)]
struct LoadInfo {
    segments: Vec<Segment>,
    pointer_sections: Vec<PointerSection>,
    /// `LC_SYMTAB` `(symoff, nsyms, stroff, strsize)`.
    symtab: Option<(u64, u32, u64, u64)>,
    /// `LC_DYSYMTAB` `(indirectsymoff, nindirectsyms)`.
    indirect: Option<(u64, u32)>,
    /// `LC_DYLD_INFO[_ONLY]` bind streams: `(off, size)` for bind / weak / lazy.
    binds: Vec<(u64, u64)>,
    /// `LC_DYLD_CHAINED_FIXUPS` `(dataoff, datasize)`.
    chained: Option<(u64, u64)>,
    /// Ordered install-name basenames of the load-dylib commands (1-based ordinals).
    dylibs: Vec<Arc<str>>,
}

/// Walk the `ncmds` load commands of a thin slice, validating each against
/// `sizeofcmds`, and collect the metadata the import walk needs.
fn read_load_commands(view: &dyn MachOView) -> Result<LoadInfo> {
    // mach_header_64: magic(0) cputype(4) cpusubtype(8) filetype(12) ncmds(16)
    //                 sizeofcmds(20) flags(24) reserved(28).
    let magic = read_u32(view, 0)?;
    if magic != MH_MAGIC_64 {
        return Err(Error::Malformed("not a little-endian 64-bit Mach-O header"));
    }
    let ncmds = read_u32(view, 16)?;
    let sizeofcmds = u64::from(read_u32(view, 20)?);
    if ncmds > MAX_COMMANDS {
        return Err(Error::Malformed("implausible load-command count"));
    }
    let region_end = MACH_HEADER_64_SIZE
        .checked_add(sizeofcmds)
        .ok_or(Error::Malformed("load-command region overflow"))?;

    let mut info = LoadInfo::default();
    let mut cursor = MACH_HEADER_64_SIZE;
    for _ in 0..ncmds {
        // Every command must fit fully inside [header_end, header_end+sizeofcmds).
        if cursor
            .checked_add(8)
            .ok_or(Error::Malformed("command header overflow"))?
            > region_end
        {
            return Err(Error::Malformed("load command runs past sizeofcmds"));
        }
        let cmd = read_u32(view, cursor)?;
        let cmdsize = u64::from(read_u32(view, cursor + 4)?);
        // A zero or unaligned cmdsize would loop forever / desync the stream.
        if cmdsize < 8 || !cmdsize.is_multiple_of(8) {
            return Err(Error::Malformed("invalid load-command size"));
        }
        let next = cursor
            .checked_add(cmdsize)
            .ok_or(Error::Malformed("command size overflow"))?;
        if next > region_end {
            return Err(Error::Malformed("load command extends past sizeofcmds"));
        }
        parse_one_command(view, cmd, cursor, cmdsize, &mut info)?;
        cursor = next;
    }
    Ok(info)
}

/// Decode a single load command at file offset `base` (already bounds-checked to
/// span `cmdsize` bytes within the command region) into `info`.
fn parse_one_command(
    view: &dyn MachOView,
    cmd: u32,
    base: u64,
    cmdsize: u64,
    info: &mut LoadInfo,
) -> Result<()> {
    match cmd {
        LC_SEGMENT_64 => read_segment(view, base, cmdsize, info),
        LC_SYMTAB => {
            // symtab_command: cmd(0) cmdsize(4) symoff(8) nsyms(12) stroff(16) strsize(20).
            if cmdsize < 24 {
                return Err(Error::Malformed("truncated LC_SYMTAB"));
            }
            let symoff = u64::from(read_u32(view, base + 8)?);
            let nsyms = read_u32(view, base + 12)?;
            let stroff = u64::from(read_u32(view, base + 16)?);
            let strsize = u64::from(read_u32(view, base + 20)?);
            info.symtab = Some((symoff, nsyms, stroff, strsize));
            Ok(())
        }
        LC_DYSYMTAB => {
            // dysymtab_command: indirectsymoff(56) nindirectsyms(60).
            if cmdsize < 80 {
                return Err(Error::Malformed("truncated LC_DYSYMTAB"));
            }
            let indirectsymoff = u64::from(read_u32(view, base + 56)?);
            let nindirectsyms = read_u32(view, base + 60)?;
            info.indirect = Some((indirectsymoff, nindirectsyms));
            Ok(())
        }
        LC_DYLD_INFO | LC_DYLD_INFO_ONLY => {
            // dyld_info_command: bind(16,20) weak_bind(24,28) lazy_bind(32,36).
            if cmdsize < 48 {
                return Err(Error::Malformed("truncated LC_DYLD_INFO"));
            }
            for (off_field, size_field) in [(16u64, 20u64), (24, 28), (32, 36)] {
                let off = u64::from(read_u32(view, base + off_field)?);
                let size = u64::from(read_u32(view, base + size_field)?);
                if size != 0 {
                    info.binds.push((off, size));
                }
            }
            Ok(())
        }
        LC_DYLD_CHAINED_FIXUPS => {
            // linkedit_data_command: dataoff(8) datasize(12).
            if cmdsize < 16 {
                return Err(Error::Malformed("truncated LC_DYLD_CHAINED_FIXUPS"));
            }
            let dataoff = u64::from(read_u32(view, base + 8)?);
            let datasize = u64::from(read_u32(view, base + 12)?);
            info.chained = Some((dataoff, datasize));
            Ok(())
        }
        LC_LOAD_DYLIB | LC_LOAD_WEAK_DYLIB | LC_REEXPORT_DYLIB | LC_LOAD_UPWARD_DYLIB
        | LC_LAZY_LOAD_DYLIB => {
            // dylib_command: dylib.name string offset at (8); string within [base, base+cmdsize).
            if info.dylibs.len() >= MAX_DYLIBS {
                return Ok(()); // ignore an absurd dylib list rather than allocate forever
            }
            let name_off = u64::from(read_u32(view, base + 8)?);
            let name = if name_off < cmdsize {
                read_cstr(view, base + name_off).unwrap_or_else(|_| "(unknown-dylib)".to_owned())
            } else {
                "(unknown-dylib)".to_owned()
            };
            info.dylibs.push(basename(&name).into());
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Parse one `LC_SEGMENT_64` and its `section_64` array.
fn read_segment(view: &dyn MachOView, base: u64, cmdsize: u64, info: &mut LoadInfo) -> Result<()> {
    if cmdsize < SEGMENT_COMMAND_64_SIZE {
        return Err(Error::Malformed("truncated LC_SEGMENT_64"));
    }
    // segment_command_64: vmaddr(24) vmsize(32) fileoff(40) filesize(48) nsects(64).
    let vmaddr = read_u64(view, base + 24)?;
    let vmsize = read_u64(view, base + 32)?;
    let fileoff = read_u64(view, base + 40)?;
    let filesize = read_u64(view, base + 48)?;
    let nsects = read_u32(view, base + 64)?;
    info.segments.push(Segment {
        vmaddr,
        vmsize,
        fileoff,
        filesize,
    });

    if nsects > MAX_SECTIONS {
        return Err(Error::Malformed("implausible section count"));
    }
    // Sections must fit within the command body.
    let sections_bytes = u64::from(nsects)
        .checked_mul(SECTION_64_SIZE)
        .ok_or(Error::Malformed("section table overflow"))?;
    if SEGMENT_COMMAND_64_SIZE
        .checked_add(sections_bytes)
        .ok_or(Error::Malformed("overflow"))?
        > cmdsize
    {
        return Err(Error::Malformed(
            "section table extends past the segment command",
        ));
    }
    for i in 0..u64::from(nsects) {
        let sec = base + SEGMENT_COMMAND_64_SIZE + i * SECTION_64_SIZE;
        // section_64: addr(32) size(40) offset(48) flags(64) reserved1(68).
        let size = read_u64(view, sec + 40)?;
        let offset = u64::from(read_u32(view, sec + 48)?);
        let flags = read_u32(view, sec + 64)?;
        let reserved1 = read_u32(view, sec + 68)?;
        if matches!(
            flags & SECTION_TYPE,
            S_NON_LAZY_SYMBOL_POINTERS | S_LAZY_SYMBOL_POINTERS
        ) {
            info.pointer_sections.push(PointerSection {
                fileoff: offset,
                size,
                indirect_index: reserved1,
            });
        }
    }
    Ok(())
}

/// Final path component of a dylib install name (`/usr/lib/libSystem.B.dylib` →
/// `libSystem.B.dylib`).
fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// Resolve a two-level-namespace / bind library ordinal to an install-name
/// basename. Positive ordinals are 1-based into the load-order dylib list; special
/// (non-positive) ordinals carry no concrete provider name.
fn library_of(ordinal: i64, dylibs: &[Arc<str>]) -> Arc<str> {
    if ordinal >= 1 {
        usize::try_from(ordinal - 1)
            .ok()
            .and_then(|i| dylibs.get(i))
            .cloned()
            .unwrap_or_else(|| Arc::from(""))
    } else {
        Arc::from("")
    }
}

// ---- Source 1: the indirect symbol table (fishhook-style) ------------------

/// Enumerate imports via the indirect symbol table: for each symbol-pointer
/// section, map every slot through `reserved1 + i` into the indirect symbol table,
/// then into the symbol table and string table for the name.
fn parse_indirect(view: &dyn MachOView, info: &LoadInfo, out: &mut Vec<RawImport>) -> Result<()> {
    let (Some((symoff, nsyms, stroff, strsize)), Some((indirectoff, nindirect))) =
        (info.symtab, info.indirect)
    else {
        return Ok(()); // no symbol / indirect table — nothing to recover this way
    };

    // On arm64e the function-pointer GOT slots are authenticated; the indirect
    // path cannot see a per-slot auth bit, so flag every slot for refusal (R1).
    let authenticated = view.is_arm64e();
    for section in &info.pointer_sections {
        let count = section.size / arch::PTR_SIZE as u64;
        if count > u64::from(MAX_INDIRECT) {
            return Err(Error::Malformed(
                "implausible symbol-pointer section length",
            ));
        }
        for i in 0..count {
            // Indirect-table index for this slot.
            let indirect_idx = u64::from(section.indirect_index)
                .checked_add(i)
                .ok_or(Error::Malformed("indirect index overflow"))?;
            if indirect_idx >= u64::from(nindirect) {
                return Err(Error::Malformed("indirect index past nindirectsyms"));
            }
            let sym_index = read_u32(view, indirectoff + indirect_idx * 4)?;
            // Local / absolute sentinels carry no import name — skip.
            if sym_index & (INDIRECT_SYMBOL_LOCAL | INDIRECT_SYMBOL_ABS) != 0 {
                continue;
            }
            if u64::from(sym_index) >= u64::from(nsyms) {
                return Err(Error::Malformed("indirect symbol index past nsyms"));
            }
            // nlist_64: n_strx(0) n_type(4) n_desc(6) n_value(8).
            let nlist = symoff + u64::from(sym_index) * NLIST_64_SIZE;
            let n_strx = read_u32(view, nlist)?;
            let n_type = {
                let mut b = [0u8; 1];
                view.read(nlist + 4, &mut b)?;
                b[0]
            };
            let n_desc = read_u16(view, nlist + 6)?;
            // Only undefined external symbols are imports (skip defined / STAB).
            if n_type & N_STAB != 0 || n_type & N_TYPE != N_UNDF || n_type & N_EXT == 0 {
                continue;
            }
            let raw = read_str_in_table(view, stroff, strsize, n_strx)?;
            let Some(name) = normalize_symbol_name(&raw) else {
                continue;
            };
            if name == TLV_BOOTSTRAP {
                continue; // dyld TLS bootstrap — never a rebindable import.
            }
            let slot_off = section
                .fileoff
                .checked_add(i * arch::PTR_SIZE as u64)
                .ok_or(Error::Malformed("slot offset overflow"))?;
            let library = library_of(i64::from(n_desc >> 8 & 0xff), &info.dylibs);
            out.push(RawImport {
                library,
                symbol: Some(Symbol::Name(name)),
                version: None,
                slot: usize::try_from(view.slot_address(slot_off)?)
                    .map_err(|_| Error::Malformed("slot address overflow"))?,
                kind: ImportKind::Standard,
                authenticated,
            });
        }
    }
    Ok(())
}

// ---- Source 2: legacy LC_DYLD_INFO bind opcodes ----------------------------

/// A ULEB128 read from the opcode stream, advancing `cursor` and honoring `end`.
fn uleb128(view: &dyn MachOView, cursor: &mut u64, end: u64) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *cursor >= end {
            return Err(Error::Malformed("ULEB128 runs past the bind stream"));
        }
        let mut b = [0u8; 1];
        view.read(*cursor, &mut b)?;
        *cursor += 1;
        if shift > ULEB_MAX_SHIFT {
            return Err(Error::Malformed("ULEB128 overflow"));
        }
        result |= u64::from(b[0] & 0x7f) << shift;
        shift += 7;
        if b[0] & 0x80 == 0 {
            return Ok(result);
        }
    }
}

/// An SLEB128 read from the opcode stream (used for `SET_ADDEND_SLEB`).
fn sleb128(view: &dyn MachOView, cursor: &mut u64, end: u64) -> Result<i64> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *cursor >= end {
            return Err(Error::Malformed("SLEB128 runs past the bind stream"));
        }
        let mut b = [0u8; 1];
        view.read(*cursor, &mut b)?;
        *cursor += 1;
        if shift > ULEB_MAX_SHIFT {
            return Err(Error::Malformed("SLEB128 overflow"));
        }
        result |= i64::from(b[0] & 0x7f) << shift;
        shift += 7;
        if b[0] & 0x80 == 0 {
            if shift < 64 && b[0] & 0x40 != 0 {
                result |= -1i64 << shift;
            }
            return Ok(result);
        }
    }
}

/// The mutable state threaded through a bind-opcode stream.
struct BindState {
    seg_index: usize,
    seg_offset: u64,
    library_ordinal: i64,
    symbol: Option<String>,
}

/// Replay one bind/weak/lazy opcode stream `[off, off+size)`, emitting an import for
/// every `DO_BIND*`. This gives the slot `(segment, offset)` directly — no
fn parse_bind_stream(
    view: &dyn MachOView,
    info: &LoadInfo,
    off: u64,
    size: u64,
    out: &mut Vec<RawImport>,
) -> Result<()> {
    if size > MAX_BIND_BYTES {
        return Err(Error::Malformed("implausible bind-stream size"));
    }
    let end = off
        .checked_add(size)
        .ok_or(Error::Malformed("bind stream overflow"))?;
    let mut cursor = off;
    let mut state = BindState {
        seg_index: 0,
        seg_offset: 0,
        library_ordinal: BIND_SPECIAL_DYLIB_SELF,
        symbol: None,
    };
    while cursor < end {
        let mut b = [0u8; 1];
        view.read(cursor, &mut b)?;
        cursor += 1;
        let opcode = b[0] & BIND_OPCODE_MASK;
        let imm = b[0] & BIND_IMMEDIATE_MASK;
        match opcode {
            // Both are no-ops for enumeration: DONE terminates a sub-sequence (we
            // simply continue to `end`), SET_TYPE_IMM sets a bind type we ignore.
            BIND_OPCODE_DONE | BIND_OPCODE_SET_TYPE_IMM => {}
            BIND_OPCODE_SET_DYLIB_ORDINAL_IMM => state.library_ordinal = i64::from(imm),
            BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB => {
                state.library_ordinal =
                    i64::try_from(uleb128(view, &mut cursor, end)?).unwrap_or(i64::MAX);
            }
            BIND_OPCODE_SET_DYLIB_SPECIAL_IMM => {
                // A special ordinal: sign-extend the 4-bit immediate (0 → self, else
                // the negative BIND_SPECIAL_DYLIB_* values). No concrete provider.
                state.library_ordinal = if imm == 0 { 0 } else { i64::from(imm) | !0xf };
            }
            BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM => {
                let (name, name_bytes) = read_cstr_raw(view, cursor)?;
                // Advance past the RAW name bytes (+ its NUL) — never the lossy
                // `String` length, which inflates by 2 bytes per invalid-UTF-8 byte
                // (each becomes a 3-byte U+FFFD) and would desync the opcode stream
                // (m3).
                cursor += name_bytes + 1;
                state.symbol = normalize_symbol_name(&name);
            }
            BIND_OPCODE_SET_ADDEND_SLEB => {
                sleb128(view, &mut cursor, end)?;
            }
            BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB => {
                state.seg_index = usize::from(imm);
                state.seg_offset = uleb128(view, &mut cursor, end)?;
            }
            BIND_OPCODE_ADD_ADDR_ULEB => {
                state.seg_offset = state
                    .seg_offset
                    .wrapping_add(uleb128(view, &mut cursor, end)?);
            }
            BIND_OPCODE_DO_BIND => {
                emit_bind(view, info, &state, out)?;
                state.seg_offset = state.seg_offset.wrapping_add(arch::PTR_SIZE as u64);
            }
            BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB => {
                emit_bind(view, info, &state, out)?;
                let add = uleb128(view, &mut cursor, end)?;
                state.seg_offset = state
                    .seg_offset
                    .wrapping_add(add)
                    .wrapping_add(arch::PTR_SIZE as u64);
            }
            BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED => {
                emit_bind(view, info, &state, out)?;
                let scaled = u64::from(imm).wrapping_mul(arch::PTR_SIZE as u64);
                state.seg_offset = state
                    .seg_offset
                    .wrapping_add(scaled)
                    .wrapping_add(arch::PTR_SIZE as u64);
            }
            BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB => {
                let count = uleb128(view, &mut cursor, end)?;
                let skip = uleb128(view, &mut cursor, end)?;
                if count > MAX_CHAIN_STEPS {
                    return Err(Error::Malformed("implausible bind repeat count"));
                }
                for _ in 0..count {
                    emit_bind(view, info, &state, out)?;
                    state.seg_offset = state
                        .seg_offset
                        .wrapping_add(skip)
                        .wrapping_add(arch::PTR_SIZE as u64);
                }
            }
            _ => return Err(Error::Malformed("unknown bind opcode")),
        }
    }
    Ok(())
}

/// Emit one import for a `DO_BIND` in the current [`BindState`].
fn emit_bind(
    view: &dyn MachOView,
    info: &LoadInfo,
    state: &BindState,
    out: &mut Vec<RawImport>,
) -> Result<()> {
    let Some(name) = state.symbol.clone() else {
        return Err(Error::Malformed("DO_BIND with no symbol set"));
    };
    if name == TLV_BOOTSTRAP {
        // dyld TLS bootstrap — bound like any other symbol, but never surfaced as a
        // rebindable import. The caller still advances `seg_offset` past this slot.
        return Ok(());
    }
    let segment = info
        .segments
        .get(state.seg_index)
        .ok_or(Error::Malformed("bind segment index out of range"))?;
    let slot_off = segment
        .fileoff
        .checked_add(state.seg_offset)
        .ok_or(Error::Malformed("bind slot offset overflow"))?;
    out.push(RawImport {
        library: library_of(state.library_ordinal, &info.dylibs),
        symbol: Some(Symbol::Name(name)),
        version: None,
        slot: usize::try_from(view.slot_address(slot_off)?)
            .map_err(|_| Error::Malformed("slot address overflow"))?,
        kind: ImportKind::Standard,
        authenticated: false,
    });
    Ok(())
}

// ---- Source 3: chained fixups ----------------------------------------------

/// One decoded fixup-chain slot.
struct ChainEntry {
    /// `true` for a bind (imports a symbol), `false` for a rebase.
    is_bind: bool,
    /// Import ordinal into the chained-fixups imports array (bind only).
    ordinal: u32,
    /// `true` if this is an arm64e **authenticated** (PAC-signed) slot.
    authenticated: bool,
    /// Steps to the next slot, in bytes (0 ends the chain in this page).
    next: u64,
}

/// Decode a raw 64-bit chain value under `format` into a [`ChainEntry`], or `None`
/// for a pointer format this engine does not walk.
fn decode_chain(raw: u64, format: u16) -> Option<ChainEntry> {
    match format {
        DYLD_CHAINED_PTR_64 | DYLD_CHAINED_PTR_64_OFFSET => {
            let is_bind = raw >> 63 & 1 == 1;
            let next = (raw >> 51 & 0xfff) * 4; // 12-bit next, 4-byte stride
            let ordinal = if is_bind { (raw & 0xff_ffff) as u32 } else { 0 };
            Some(ChainEntry {
                is_bind,
                ordinal,
                authenticated: false,
                next,
            })
        }
        DYLD_CHAINED_PTR_ARM64E
        | DYLD_CHAINED_PTR_ARM64E_KERNEL
        | DYLD_CHAINED_PTR_ARM64E_USERLAND
        | DYLD_CHAINED_PTR_ARM64E_FIRMWARE
        | DYLD_CHAINED_PTR_ARM64E_USERLAND24 => {
            let authenticated = raw >> 63 & 1 == 1;
            let is_bind = raw >> 62 & 1 == 1;
            // The 11-bit `next` field counts strides to the following slot. The
            // stride is 8 bytes for the userland/desktop arm64e formats but 4 bytes
            // for the KERNEL and FIRMWARE formats (Apple `<mach-o/fixup-chains.h>`:
            // "stride 4" vs "stride 8"). Applying a flat 8-byte stride to all of them
            // would mis-thread a KERNEL/FIRMWARE chain.
            let stride = match format {
                DYLD_CHAINED_PTR_ARM64E_KERNEL | DYLD_CHAINED_PTR_ARM64E_FIRMWARE => 4,
                _ => 8,
            };
            let next = (raw >> 51 & 0x7ff) * stride; // 11-bit next
            // USERLAND24 widens the ordinal to 24 bits; the others use 16.
            let ordinal = if format == DYLD_CHAINED_PTR_ARM64E_USERLAND24 {
                (raw & 0xff_ffff) as u32
            } else {
                (raw & 0xffff) as u32
            };
            Some(ChainEntry {
                is_bind,
                ordinal,
                authenticated,
                next,
            })
        }
        _ => None,
    }
}

/// The imports-table layout of a chained-fixups blob.
#[derive(Clone, Copy)]
struct ImportsTable {
    /// File offset of the imports array.
    base: u64,
    /// Number of imports.
    count: u32,
    /// `imports_format` (`DYLD_CHAINED_IMPORT[_ADDEND[64]]`).
    format: u32,
    /// File offset of the symbol string pool.
    symbols_base: u64,
    /// Blob end (for pool string bounds).
    blob_end: u64,
}

impl ImportsTable {
    /// Resolve the `(library_ordinal, name)` of import `ordinal`.
    fn resolve(
        &self,
        view: &dyn MachOView,
        ordinal: u32,
        dylibs: &[Arc<str>],
    ) -> Result<(Arc<str>, Option<String>)> {
        if ordinal >= self.count {
            return Err(Error::Malformed(
                "chained-fixups import ordinal out of range",
            ));
        }
        // Each import record's stride is implied by the format (4 / 8 / 16 bytes);
        // the leading bitfield word carries the library ordinal and name offset.
        let (name_off, lib_ordinal) = match self.format {
            DYLD_CHAINED_IMPORT => {
                let word = read_u32(view, self.base + u64::from(ordinal) * 4)?;
                (
                    u64::from(word >> 9 & 0x7f_ffff),
                    sign_extend(u64::from(word & 0xff), 8),
                )
            }
            DYLD_CHAINED_IMPORT_ADDEND => {
                let word = read_u32(view, self.base + u64::from(ordinal) * 8)?;
                (
                    u64::from(word >> 9 & 0x7f_ffff),
                    sign_extend(u64::from(word & 0xff), 8),
                )
            }
            DYLD_CHAINED_IMPORT_ADDEND64 => {
                let word = read_u64(view, self.base + u64::from(ordinal) * 16)?;
                (word >> 32 & 0xffff_ffff, sign_extend(word & 0xffff, 16))
            }
            _ => {
                return Err(Error::Malformed(
                    "unsupported chained-fixups imports_format",
                ));
            }
        };
        let name_addr = self
            .symbols_base
            .checked_add(name_off)
            .ok_or(Error::Malformed("chained-fixups name offset overflow"))?;
        if name_addr >= self.blob_end {
            return Err(Error::Malformed("chained-fixups name offset past blob"));
        }
        let name = read_cstr(view, name_addr)?;
        Ok((
            library_of(lib_ordinal, dylibs),
            normalize_symbol_name(&name),
        ))
    }
}

/// Sign-extend the low `bits` bits of `value` to `i64` (bind library ordinals are
/// signed: the top values are the negative `BIND_SPECIAL_DYLIB_*`).
fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    // Reinterpret the shifted bits as signed, then arithmetic-shift back.
    i64::from_ne_bytes((value << shift).to_ne_bytes()) >> shift
}

/// Parse `LC_DYLD_CHAINED_FIXUPS` and emit an import for every bind slot. Chain
/// *metadata* is read through `view.read`; the per-slot chain encodings are read
/// through `view.read_slot_encoding` (the on-disk file on a live image), so this
/// runs only when the view is [`file_backed`](MachOView::file_backed).
fn parse_chained_fixups(
    view: &dyn MachOView,
    info: &LoadInfo,
    dataoff: u64,
    datasize: u64,
    out: &mut Vec<RawImport>,
) -> Result<()> {
    if !view.file_backed() {
        // Live image with no on-disk bytes: the in-memory slots are already
        // fixed-up, so the chain cannot be walked. The indirect-symbol path covers
        // the enumeration.
        return Ok(());
    }
    let blob_end = dataoff
        .checked_add(datasize)
        .ok_or(Error::Malformed("chained blob overflow"))?;
    // dyld_chained_fixups_header: fixups_version(0) starts_offset(4) imports_offset(8)
    //                             symbols_offset(12) imports_count(16) imports_format(20).
    if datasize < 28 {
        return Err(Error::Malformed("truncated chained-fixups header"));
    }
    let fixups_version = read_u32(view, dataoff)?;
    if fixups_version != 0 {
        return Err(Error::Unsupported("unknown chained-fixups version"));
    }
    let starts_offset = u64::from(read_u32(view, dataoff + 4)?);
    let imports_offset = u64::from(read_u32(view, dataoff + 8)?);
    let symbols_offset = u64::from(read_u32(view, dataoff + 12)?);
    let imports_count = read_u32(view, dataoff + 16)?;
    let imports_format = read_u32(view, dataoff + 20)?;

    let imports = ImportsTable {
        base: dataoff
            .checked_add(imports_offset)
            .ok_or(Error::Malformed("imports off overflow"))?,
        count: imports_count,
        format: imports_format,
        symbols_base: dataoff
            .checked_add(symbols_offset)
            .ok_or(Error::Malformed("symbols off overflow"))?,
        blob_end,
    };

    let starts = dataoff
        .checked_add(starts_offset)
        .ok_or(Error::Malformed("starts off overflow"))?;
    // dyld_chained_starts_in_image: seg_count(0) seg_info_offset[seg_count](4..).
    let seg_count = read_u32(view, starts)?;
    if seg_count > MAX_FIXUP_SEGMENTS {
        return Err(Error::Malformed("implausible fixup segment count"));
    }
    for i in 0..u64::from(seg_count) {
        let seg_info_offset = u64::from(read_u32(view, starts + 4 + i * 4)?);
        if seg_info_offset == 0 {
            continue; // no fixups in this segment
        }
        let seg_starts = starts
            .checked_add(seg_info_offset)
            .ok_or(Error::Malformed("segment starts overflow"))?;
        walk_segment_chains(view, info, &imports, seg_starts, out)?;
    }
    Ok(())
}

/// Walk every page chain of one `dyld_chained_starts_in_segment`.
fn walk_segment_chains(
    view: &dyn MachOView,
    info: &LoadInfo,
    imports: &ImportsTable,
    seg_starts: u64,
    out: &mut Vec<RawImport>,
) -> Result<()> {
    // dyld_chained_starts_in_segment: size(0) page_size(4) pointer_format(6)
    //   segment_offset(8) max_valid_pointer(16) page_count(20) page_start[](22..).
    let page_size = u64::from(read_u16(view, seg_starts + 4)?);
    let pointer_format = read_u16(view, seg_starts + 6)?;
    let segment_offset = read_u64(view, seg_starts + 8)?;
    let page_count = read_u16(view, seg_starts + 20)?;
    if page_size == 0 {
        return Err(Error::Malformed("zero fixup page size"));
    }
    for j in 0..u64::from(page_count) {
        let page_start = read_u16(view, seg_starts + 22 + j * 2)?;
        if page_start == DYLD_CHAINED_PTR_START_NONE {
            continue;
        }
        // First fixup's file offset in this page.
        let mut offset = segment_offset
            .checked_add(j * page_size)
            .and_then(|v| v.checked_add(u64::from(page_start)))
            .ok_or(Error::Malformed("fixup page offset overflow"))?;
        // Follow the chain within the page.
        for _ in 0..MAX_CHAIN_STEPS {
            let raw = view.read_slot_encoding(offset)?;
            let Some(entry) = decode_chain(raw, pointer_format) else {
                return Err(Error::Unsupported("unsupported chained-pointer format"));
            };
            if entry.is_bind {
                let (library, name) = imports.resolve(view, entry.ordinal, &info.dylibs)?;
                // Skip the dyld TLS bootstrap slot (never a rebindable import).
                if let Some(name) = name.filter(|n| n != TLV_BOOTSTRAP) {
                    out.push(RawImport {
                        library,
                        symbol: Some(Symbol::Name(name)),
                        version: None,
                        slot: usize::try_from(view.slot_address(offset)?)
                            .map_err(|_| Error::Malformed("slot address overflow"))?,
                        kind: ImportKind::Standard,
                        authenticated: entry.authenticated,
                    });
                }
            }
            if entry.next == 0 {
                break;
            }
            offset = offset
                .checked_add(entry.next)
                .ok_or(Error::Malformed("fixup chain offset overflow"))?;
        }
    }
    Ok(())
}

// ---- Public entry points ---------------------------------------------------

/// Recover just the `LC_SEGMENT_64` table of a thin 64-bit Mach-O image behind
/// `view` — the bootstrap the live Darwin engine ([`crate::module`], macOS + iOS)
/// uses to build its file-offset ↔ address translation before the full view exists.
#[cfg_attr(not(any(target_os = "macos", target_os = "ios")), allow(dead_code))]
pub(crate) fn segments_of(view: &dyn MachOView) -> Result<Vec<Segment>> {
    Ok(read_load_commands(view)?.segments)
}

/// Enumerate every rebindable import slot of a thin 64-bit Mach-O image behind
/// `view`, merging the indirect-symbol, legacy-bind, and chained-fixups sources and
pub(crate) fn parse_imports(view: &dyn MachOView) -> Result<Vec<RawImport>> {
    if !arch::macho_cpu_supported(view.cpu_type()) {
        return Err(Error::Unsupported("Mach-O cputype is not x86-64 or arm64"));
    }
    let info = read_load_commands(view)?;

    // Source 1 (PRIMARY): the indirect symbol table. It is pure `__LINKEDIT`
    // metadata, self-sufficient, and readable from the live in-memory image
    // (fishhook-style), so it is the authoritative enumeration. A failure here is a
    // genuine structural defect in the symbol / indirect / string tables, so it
    // propagates.
    let mut out = Vec::new();
    parse_indirect(view, &info, &mut out)?;

    // Sources 2/3 (COMPLEMENT): chained fixups XOR the legacy `LC_DYLD_INFO` bind
    // opcodes (an image carries at most one). They refine the primary set — reaching
    // bind slots outside the symbol-pointer sections and, for chained fixups,
    // carrying the PRECISE per-slot arm64e authentication bit — but they are not
    // required: the indirect table already stands on its own. Parse them into a
    // separate buffer and adopt it only if the whole secondary parse succeeds, so an
    // `Error::Unsupported` pointer format or a malformed secondary blob degrades to
    // "primary-only" instead of discarding the already-recovered primary enumeration
    // (M3). Every read is bounds-checked regardless, so this never trades away memory
    // safety.
    let mut complement = Vec::new();
    let complement_result = if let Some((dataoff, datasize)) = info.chained {
        parse_chained_fixups(view, &info, dataoff, datasize, &mut complement)
    } else {
        info.binds
            .iter()
            .try_for_each(|&(off, size)| parse_bind_stream(view, &info, off, size, &mut complement))
    };
    if complement_result.is_ok() {
        merge_complement(&mut out, complement);
    }

    // Guarantee a unique slot set even if a malformed image mapped two file offsets
    // onto one address (keep first).
    let mut seen: HashSet<usize> = HashSet::new();
    out.retain(|import| seen.insert(import.slot));
    Ok(out)
}

/// Merge the best-effort complement enumeration (chained fixups or legacy binds)
/// into the primary indirect set `out`, deduplicating by slot address.
///
/// Where a slot is enumerated by BOTH sources, the primary record is kept but adopts
/// the complement's **precise** per-slot `authenticated` flag. The indirect path can
/// only apply the coarse whole-image arm64e flag ([`MachOView::is_arm64e`]), whereas
/// the chain walk decodes the exact per-slot arm64e `auth` bit — so a genuinely
/// authenticated `__auth_got` slot stays refused (R1) while a non-authenticated
/// `__got` slot on an arm64e image becomes rebindable (m2). Slots seen only by the
/// complement are appended.
fn merge_complement(out: &mut Vec<RawImport>, complement: Vec<RawImport>) {
    let mut slot_to_index: HashMap<usize, usize> = HashMap::with_capacity(out.len());
    for (i, import) in out.iter().enumerate() {
        slot_to_index.entry(import.slot).or_insert(i);
    }
    for import in complement {
        if let Some(&i) = slot_to_index.get(&import.slot) {
            // Shared slot: keep the primary record, take the precise auth truth.
            if let Some(existing) = out.get_mut(i) {
                existing.authenticated = import.authenticated;
            }
        } else {
            slot_to_index.insert(import.slot, out.len());
            out.push(import);
        }
    }
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

    /// Base vmaddr of the synthetic `__TEXT` segment (nonzero, so the live-vs-file
    /// `vmaddr != fileoff` translation is exercised for `slot_address`).
    const TEXT_VMADDR: u64 = 0x1_0000_0000;
    /// The `__DATA` segment maps at this vmaddr; its file offset differs from it.
    const DATA_VMADDR: u64 = 0x1_0001_0000;
    /// File offset of the `__DATA` segment in the synthetic image.
    const DATA_FILEOFF: u64 = 0x4000;

    /// An offline [`MachOView`] over a borrowed byte slice starting at `base` (a fat
    /// slice offset, or 0 for a thin file). File offsets are relative to `base`.
    struct FileView<'a> {
        bytes: &'a [u8],
        base: u64,
        segments: Vec<Segment>,
        cpu_type: u32,
        cpu_subtype: u32,
    }

    impl<'a> FileView<'a> {
        /// Build a view over the thin slice at `base`, pre-reading the segment table
        /// so `slot_address` can translate a file offset to its link-time vmaddr.
        fn new(bytes: &'a [u8], base: u64) -> Result<Self> {
            let head = base.checked_add(8).ok_or(Error::Malformed("overflow"))? as usize;
            let magic_bytes = bytes
                .get(base as usize..head)
                .ok_or(Error::Malformed("slice too small"))?;
            if u32::from_le_bytes(magic_bytes[0..4].try_into().unwrap()) != MH_MAGIC_64 {
                return Err(Error::Malformed("not MH_MAGIC_64"));
            }
            let cpu_type = u32::from_le_bytes(
                bytes
                    .get(base as usize + 4..base as usize + 8)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            );
            let cpu_subtype = u32::from_le_bytes(
                bytes
                    .get(base as usize + 8..base as usize + 12)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            );
            // A bootstrap pass reads the load commands to collect the segments (the
            // real parser reads them again through the finished view — cheap).
            let mut boot = Self {
                bytes,
                base,
                segments: Vec::new(),
                cpu_type,
                cpu_subtype,
            };
            let info = read_load_commands(&boot)?;
            boot.segments = info.segments;
            Ok(boot)
        }
    }

    impl MachOView for FileView<'_> {
        fn cpu_type(&self) -> u32 {
            self.cpu_type
        }
        fn cpu_subtype(&self) -> u32 {
            self.cpu_subtype
        }
        fn read(&self, off: u64, buf: &mut [u8]) -> Result<()> {
            let start = self
                .base
                .checked_add(off)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(Error::Malformed("cursor overflow"))?;
            let end = start
                .checked_add(buf.len())
                .ok_or(Error::Malformed("read overflow"))?;
            let src = self
                .bytes
                .get(start..end)
                .ok_or(Error::Malformed("read past slice"))?;
            buf.copy_from_slice(src);
            Ok(())
        }
        fn file_backed(&self) -> bool {
            true
        }
        fn read_slot_encoding(&self, off: u64) -> Result<u64> {
            // Offline: the encoded pointer is in the same buffer.
            let mut b = [0u8; 8];
            self.read(off, &mut b)?;
            Ok(u64::from_le_bytes(b))
        }
        fn slot_address(&self, off: u64) -> Result<u64> {
            let seg = self
                .segments
                .iter()
                .find(|s| s.contains_fileoff(off))
                .ok_or(Error::Malformed("slot offset outside any segment"))?;
            let addr = seg
                .vmaddr
                .checked_add(off - seg.fileoff)
                .ok_or(Error::Malformed("slot vmaddr overflow"))?;
            if !addr.is_multiple_of(8) {
                return Err(Error::Malformed("slot not pointer-aligned"));
            }
            Ok(addr)
        }
    }

    /// A little-endian byte assembler over a fixed-size image, with patch-back.
    struct Image {
        bytes: Vec<u8>,
    }
    impl Image {
        fn new(size: usize) -> Self {
            Self {
                bytes: vec![0u8; size],
            }
        }
        fn put_u16(&mut self, at: usize, v: u16) {
            self.bytes[at..at + 2].copy_from_slice(&v.to_le_bytes());
        }
        fn put_u32(&mut self, at: usize, v: u32) {
            self.bytes[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        fn put_u64(&mut self, at: usize, v: u64) {
            self.bytes[at..at + 8].copy_from_slice(&v.to_le_bytes());
        }
        fn put_str(&mut self, at: usize, s: &str) {
            self.bytes[at..at + s.len()].copy_from_slice(s.as_bytes());
            self.bytes[at + s.len()] = 0;
        }
    }

    /// The layout knobs the two builders share.
    struct Layout {
        cpu_type: u32,
        cpu_subtype: u32,
    }

    /// Build a synthetic thin arm64 (or x86-64) Mach-O whose imports are recoverable
    /// via the **indirect symbol table**: a `__DATA,__la_symbol_ptr` section with
    /// `reserved1` into an indirect table → symtab → strtab.
    ///
    /// Returns `(bytes, [(name, slot_vmaddr)])`.
    fn build_indirect(layout: &Layout, names: &[&str]) -> (Vec<u8>, Vec<(String, u64)>) {
        let n = names.len();
        // File map:
        //   0x0000 header + load commands
        //   0x4000 __DATA: la_symbol_ptr slots (8 bytes each)
        //   0x5000 indirect symbol table (u32 each)
        //   0x6000 symbol table (nlist_64, 16 bytes each)
        //   0x7000 string table
        let ptrs_off = DATA_FILEOFF as usize;
        let indirect_off = 0x5000usize;
        let symtab_off = 0x6000usize;
        let strtab_off = 0x7000usize;
        let mut img = Image::new(0x8000);

        // Header.
        img.put_u32(0, MH_MAGIC_64);
        img.put_u32(4, layout.cpu_type);
        img.put_u32(8, layout.cpu_subtype);
        img.put_u32(12, 0x6); // MH_DYLIB
        // ncmds / sizeofcmds patched after building the commands.

        // Load commands start at 32.
        let mut c = 32usize;
        let mut ncmds = 0u32;

        // LC_SEGMENT_64 __TEXT (covers the header region at fileoff 0).
        let text = c;
        img.put_u32(text, LC_SEGMENT_64);
        img.put_u32(text + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(text + 8, "__TEXT");
        img.put_u64(text + 24, TEXT_VMADDR); // vmaddr
        img.put_u64(text + 32, 0x4000); // vmsize
        img.put_u64(text + 40, 0); // fileoff
        img.put_u64(text + 48, 0x4000); // filesize
        img.put_u32(text + 64, 0); // nsects
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // LC_SEGMENT_64 __DATA with one la_symbol_ptr section.
        let data = c;
        let seg_and_one_sec = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
        img.put_u32(data, LC_SEGMENT_64);
        img.put_u32(data + 4, seg_and_one_sec as u32);
        img.put_str(data + 8, "__DATA");
        img.put_u64(data + 24, DATA_VMADDR); // vmaddr
        img.put_u64(data + 32, 0x1000); // vmsize
        img.put_u64(data + 40, DATA_FILEOFF); // fileoff
        img.put_u64(data + 48, 0x1000); // filesize
        img.put_u32(data + 64, 1); // nsects
        // section_64 __la_symbol_ptr.
        let sec = data + SEGMENT_COMMAND_64_SIZE as usize;
        img.put_str(sec, "__la_symbol_ptr");
        img.put_str(sec + 16, "__DATA");
        img.put_u64(sec + 32, DATA_VMADDR); // addr
        img.put_u64(sec + 40, (n * 8) as u64); // size
        img.put_u32(sec + 48, ptrs_off as u32); // offset
        img.put_u32(sec + 64, S_LAZY_SYMBOL_POINTERS); // flags
        img.put_u32(sec + 68, 0); // reserved1 = indirect base index 0
        c += seg_and_one_sec as usize;
        ncmds += 1;

        // LC_SEGMENT_64 __LINKEDIT (covers indirect/symtab/strtab).
        let link = c;
        img.put_u32(link, LC_SEGMENT_64);
        img.put_u32(link + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(link + 8, "__LINKEDIT");
        img.put_u64(link + 24, DATA_VMADDR + 0x1000); // vmaddr
        img.put_u64(link + 32, 0x4000); // vmsize
        img.put_u64(link + 40, indirect_off as u64); // fileoff
        img.put_u64(link + 48, 0x3000); // filesize
        img.put_u32(link + 64, 0); // nsects
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // LC_LOAD_DYLIB (ordinal 1).
        let dylib = c;
        let dylib_size = (24 + "libSystem.B.dylib".len() + 1).next_multiple_of(8); // padded
        img.put_u32(dylib, LC_LOAD_DYLIB);
        img.put_u32(dylib + 4, dylib_size as u32);
        img.put_u32(dylib + 8, 24); // name offset
        img.put_str(dylib + 24, "libSystem.B.dylib");
        c += dylib_size;
        ncmds += 1;

        // LC_SYMTAB.
        let symtab = c;
        img.put_u32(symtab, LC_SYMTAB);
        img.put_u32(symtab + 4, 24);
        img.put_u32(symtab + 8, symtab_off as u32); // symoff
        img.put_u32(symtab + 12, n as u32); // nsyms
        img.put_u32(symtab + 16, strtab_off as u32); // stroff
        img.put_u32(symtab + 20, (0x8000 - strtab_off) as u32); // strsize
        c += 24;
        ncmds += 1;

        // LC_DYSYMTAB (only indirectsymoff/nindirectsyms are read).
        let dysym = c;
        img.put_u32(dysym, LC_DYSYMTAB);
        img.put_u32(dysym + 4, 80);
        img.put_u32(dysym + 56, indirect_off as u32); // indirectsymoff
        img.put_u32(dysym + 60, n as u32); // nindirectsyms
        c += 80;
        ncmds += 1;

        img.put_u32(16, ncmds);
        img.put_u32(20, (c - 32) as u32); // sizeofcmds

        // Indirect table, symbol table, string table, and the pointer slots.
        let mut strx = 1u32; // index 0 is the leading NUL
        let mut expected = Vec::new();
        for (i, name) in names.iter().enumerate() {
            // Mach-O symbol names carry a leading underscore.
            let mangled = format!("_{name}");
            img.put_u32(indirect_off + i * 4, i as u32); // indirect[i] = symbol index i
            let nl = symtab_off + i * 16;
            img.put_u32(nl, strx); // n_strx
            img.bytes[nl + 4] = N_UNDF | N_EXT; // n_type: undefined external
            img.put_u16(nl + 6, 1 << 8); // n_desc: library ordinal 1
            img.put_str(strtab_off + strx as usize, &mangled);
            strx += mangled.len() as u32 + 1;
            expected.push((
                normalize_symbol_name(&mangled).unwrap(),
                DATA_VMADDR + (i * 8) as u64,
            ));
        }
        (img.bytes, expected)
    }

    fn find<'a>(imports: &'a [RawImport], name: &str) -> Option<&'a RawImport> {
        imports
            .iter()
            .find(|i| i.symbol == Some(Symbol::name(name)))
    }

    #[test]
    fn indirect_symbol_table_thin_arm64() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: 0,
        };
        let (bytes, expected) = build_indirect(&layout, &["open", "close", "stat"]);
        let view = FileView::new(&bytes, 0).expect("valid synthetic thin Mach-O");
        assert_eq!(view.cpu_type(), arch::CPU_TYPE_ARM64);
        let imports = parse_imports(&view).expect("enumerate indirect-symbol imports");

        for (name, slot) in &expected {
            let import = find(&imports, name).unwrap_or_else(|| panic!("missing import {name}"));
            assert_eq!(import.slot as u64, *slot, "slot vmaddr for {name}");
            assert!(!import.authenticated);
            assert_eq!(&*import.library, "libSystem.B.dylib");
        }
        assert_eq!(imports.len(), 3);
    }

    #[test]
    fn leading_underscore_and_inode64_suffix_are_normalized() {
        assert_eq!(
            normalize_symbol_name("_stat$INODE64").as_deref(),
            Some("stat")
        );
        assert_eq!(normalize_symbol_name("_open").as_deref(), Some("open"));
        assert_eq!(
            normalize_symbol_name("_readdir$INODE64").as_deref(),
            Some("readdir")
        );
        assert_eq!(
            normalize_symbol_name("___getdirentries64").as_deref(),
            Some("__getdirentries64")
        );
        assert_eq!(normalize_symbol_name("_").as_deref(), None);
        assert_eq!(normalize_symbol_name("").as_deref(), None);

        // End-to-end: a slice importing the x86-64 `$INODE64` alias matches `stat`.
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let (bytes, _) = build_indirect(&layout, &["stat$INODE64", "open"]);
        let view = FileView::new(&bytes, 0).unwrap();
        let imports = parse_imports(&view).unwrap();
        assert!(
            find(&imports, "stat").is_some(),
            "stat$INODE64 normalizes to stat"
        );
        assert!(find(&imports, "open").is_some());
    }

    #[test]
    fn tlv_bootstrap_is_hidden_from_indirect_enumeration() {
        // `__tlv_bootstrap` (dyld's TLS bootstrap) is bound like any import but must
        // never be surfaced as a rebindable slot — rebinding it would corrupt TLS
        // setup (`plthook_osx` hides it too). The build helper prepends one '_', so
        // "_tlv_bootstrap" produces the raw Mach-O symbol "__tlv_bootstrap" (which
        // normalizes to "_tlv_bootstrap" == `TLV_BOOTSTRAP`).
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: 0,
        };
        let (bytes, _) = build_indirect(&layout, &["open", "_tlv_bootstrap", "close"]);
        let view = FileView::new(&bytes, 0).expect("valid synthetic thin Mach-O");
        let imports = parse_imports(&view).expect("enumerate indirect imports");
        assert!(find(&imports, "open").is_some());
        assert!(find(&imports, "close").is_some());
        assert!(
            find(&imports, "_tlv_bootstrap").is_none(),
            "TLS bootstrap must be hidden"
        );
        assert_eq!(imports.len(), 2, "only the two real imports remain");
    }

    #[test]
    fn tlv_bootstrap_is_hidden_from_chained_enumeration() {
        // The same filter applies on the chained-fixups path.
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let (bytes, _) = build_chained(
            &layout,
            DYLD_CHAINED_PTR_64_OFFSET,
            DYLD_CHAINED_IMPORT,
            &["open", "_tlv_bootstrap", "close"],
        );
        let view = FileView::new(&bytes, 0).expect("valid chained-fixups image");
        let imports = parse_imports(&view).expect("enumerate chained binds");
        assert!(find(&imports, "open").is_some());
        assert!(find(&imports, "close").is_some());
        assert!(
            find(&imports, "_tlv_bootstrap").is_none(),
            "TLS bootstrap must be hidden"
        );
        assert_eq!(imports.len(), 2);
    }

    /// A universal (fat) wrapper around the thin arm64 slice parses the selected
    /// slice via its slice offset.
    #[test]
    fn fat_wrapper_selects_slice() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: 0,
        };
        let (thin, expected) = build_indirect(&layout, &["open", "read"]);

        // Assemble a single-arch fat container (big-endian header).
        let slice_off = 0x4000usize;
        let mut fat = vec![0u8; slice_off + thin.len()];
        fat[0..4].copy_from_slice(&FAT_MAGIC.to_be_bytes());
        fat[4..8].copy_from_slice(&1u32.to_be_bytes()); // nfat_arch
        fat[8..12].copy_from_slice(&arch::CPU_TYPE_ARM64.to_be_bytes());
        fat[12..16].copy_from_slice(&0u32.to_be_bytes()); // cpusubtype
        fat[16..20].copy_from_slice(&(slice_off as u32).to_be_bytes()); // offset
        fat[20..24].copy_from_slice(&(thin.len() as u32).to_be_bytes()); // size
        fat[24..28].copy_from_slice(&14u32.to_be_bytes()); // align
        fat[slice_off..].copy_from_slice(&thin);

        let container = classify(&fat).expect("fat classify");
        let Container::Fat(slices) = container else {
            panic!("expected Fat")
        };
        assert_eq!(slices.len(), 1);
        let arch_slice = slices[0];
        assert_eq!(arch_slice.cpu_type, arch::CPU_TYPE_ARM64);
        assert_eq!(arch_slice.range.offset, slice_off as u64);
        assert_eq!(arch_slice.range.size, thin.len() as u64);

        let view = FileView::new(&fat, arch_slice.range.offset).expect("thin slice view");
        let imports = parse_imports(&view).expect("enumerate fat slice imports");
        for (name, slot) in &expected {
            assert_eq!(find(&imports, name).unwrap().slot as u64, *slot);
        }
    }

    /// Encode one **bind** fixup for `pointer_format`, matching Apple
    /// `<mach-o/fixup-chains.h>`. The slot is assumed to sit 8 bytes before the next
    /// one; `is_last` terminates the chain (`next = 0`). `authenticated` sets the
    /// arm64e per-slot auth bit (the non-arm64e formats ignore it).
    fn encode_chain_bind(
        pointer_format: u16,
        ordinal: u64,
        authenticated: bool,
        is_last: bool,
    ) -> u64 {
        match pointer_format {
            DYLD_CHAINED_PTR_64 | DYLD_CHAINED_PTR_64_OFFSET => {
                // ordinal:24, addend:8, reserved:19, next:12, bind:1. Slots are 8
                // bytes apart at a 4-byte stride, so `next` = 2 (0 ends the chain).
                let next: u64 = if is_last { 0 } else { 2 };
                (1u64 << 63) | (next << 51) | (ordinal & 0xff_ffff)
            }
            DYLD_CHAINED_PTR_ARM64E
            | DYLD_CHAINED_PTR_ARM64E_USERLAND
            | DYLD_CHAINED_PTR_ARM64E_USERLAND24 => {
                // ordinal:16|24, ..., next:11, bind:1, auth:1. Slots are 8 bytes apart
                // at an 8-byte stride, so `next` = 1 (0 ends the chain).
                let next: u64 = u64::from(!is_last);
                let ord_mask = if pointer_format == DYLD_CHAINED_PTR_ARM64E_USERLAND24 {
                    0xff_ffff
                } else {
                    0xffff
                };
                (u64::from(authenticated) << 63)
                    | (1u64 << 62)
                    | (next << 51)
                    | (ordinal & ord_mask)
            }
            other => panic!("encode_chain_bind: unsupported pointer format {other}"),
        }
    }

    /// Write import record `index` into a chained-fixups imports array at file offset
    /// `base`, encoding `name_off` / `lib_ordinal` per `imports_format`
    /// (`DYLD_CHAINED_IMPORT[_ADDEND[64]]`, Apple `<mach-o/fixup-chains.h>`). The
    /// addend field of the `_ADDEND[64]` records is written as 0 (the parser strides
    /// over it — the addend does not affect name/library resolution).
    fn put_chained_import(
        img: &mut Image,
        base: usize,
        index: usize,
        imports_format: u32,
        name_off: u32,
        lib_ordinal: u8,
    ) {
        match imports_format {
            DYLD_CHAINED_IMPORT => {
                // lib_ordinal:8, weak_import:1, name_offset:23.
                img.put_u32(base + index * 4, (name_off << 9) | u32::from(lib_ordinal));
            }
            DYLD_CHAINED_IMPORT_ADDEND => {
                // The bitfield word (as DYLD_CHAINED_IMPORT), then an int32 addend.
                img.put_u32(base + index * 8, (name_off << 9) | u32::from(lib_ordinal));
                img.put_u32(base + index * 8 + 4, 0);
            }
            DYLD_CHAINED_IMPORT_ADDEND64 => {
                // lib_ordinal:16, weak_import:1, reserved:15, name_offset:32; then u64 addend.
                img.put_u64(
                    base + index * 16,
                    (u64::from(name_off) << 32) | u64::from(lib_ordinal),
                );
                img.put_u64(base + index * 16 + 8, 0);
            }
            other => panic!("put_chained_import: unsupported imports_format {other}"),
        }
    }

    /// Build a synthetic thin image whose imports live in a **chained-fixups** blob
    /// with a single `__DATA` page chain of `pointer_format` bind slots and an
    /// `imports_format` imports array. Returns `(bytes, [(name, slot_vmaddr)])`.
    fn build_chained(
        layout: &Layout,
        pointer_format: u16,
        imports_format: u32,
        names: &[&str],
    ) -> (Vec<u8>, Vec<(String, u64)>) {
        let n = names.len();
        let data_off = DATA_FILEOFF as usize;
        let blob_off = 0x5000usize;
        let mut img = Image::new(0x8000);

        img.put_u32(0, MH_MAGIC_64);
        img.put_u32(4, layout.cpu_type);
        img.put_u32(8, layout.cpu_subtype);
        img.put_u32(12, 0x6);

        let mut c = 32usize;
        let mut ncmds = 0u32;

        // __TEXT.
        let text = c;
        img.put_u32(text, LC_SEGMENT_64);
        img.put_u32(text + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(text + 8, "__TEXT");
        img.put_u64(text + 24, TEXT_VMADDR);
        img.put_u64(text + 32, 0x4000);
        img.put_u64(text + 40, 0);
        img.put_u64(text + 48, 0x4000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // __DATA (holds the chain slots).
        let data = c;
        img.put_u32(data, LC_SEGMENT_64);
        img.put_u32(data + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(data + 8, "__DATA");
        img.put_u64(data + 24, DATA_VMADDR);
        img.put_u64(data + 32, 0x1000);
        img.put_u64(data + 40, DATA_FILEOFF);
        img.put_u64(data + 48, 0x1000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // __LINKEDIT (holds the chained-fixups blob).
        let link = c;
        img.put_u32(link, LC_SEGMENT_64);
        img.put_u32(link + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(link + 8, "__LINKEDIT");
        img.put_u64(link + 24, DATA_VMADDR + 0x1000);
        img.put_u64(link + 32, 0x3000);
        img.put_u64(link + 40, blob_off as u64);
        img.put_u64(link + 48, 0x3000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // LC_LOAD_DYLIB (ordinal 1).
        let dylib = c;
        let dylib_size = (24 + "libSystem.B.dylib".len() + 1).next_multiple_of(8);
        img.put_u32(dylib, LC_LOAD_DYLIB);
        img.put_u32(dylib + 4, dylib_size as u32);
        img.put_u32(dylib + 8, 24);
        img.put_str(dylib + 24, "libSystem.B.dylib");
        c += dylib_size;
        ncmds += 1;

        // LC_DYLD_CHAINED_FIXUPS.
        let cf = c;
        img.put_u32(cf, LC_DYLD_CHAINED_FIXUPS);
        img.put_u32(cf + 4, 16);
        img.put_u32(cf + 8, blob_off as u32); // dataoff
        img.put_u32(cf + 12, 0x1000); // datasize
        c += 16;
        ncmds += 1;

        img.put_u32(16, ncmds);
        img.put_u32(20, (c - 32) as u32);

        // Chained-fixups blob: header, starts-in-image, one starts-in-segment,
        // imports array, symbol pool.
        // Header (28 bytes) at blob_off.
        let starts_off_rel = 0x40usize;
        let imports_off_rel = 0x100usize;
        let symbols_off_rel = 0x200usize;
        img.put_u32(blob_off, 0); // fixups_version
        img.put_u32(blob_off + 4, starts_off_rel as u32);
        img.put_u32(blob_off + 8, imports_off_rel as u32);
        img.put_u32(blob_off + 12, symbols_off_rel as u32);
        img.put_u32(blob_off + 16, n as u32); // imports_count
        img.put_u32(blob_off + 20, imports_format); // imports_format

        // dyld_chained_starts_in_image at blob_off + starts_off_rel.
        let starts = blob_off + starts_off_rel;
        img.put_u32(starts, 2); // seg_count (0=__TEXT, 1=__DATA)
        img.put_u32(starts + 4, 0); // __TEXT: no fixups
        let seg_info_rel = 0x20usize; // offset from `starts` to the starts_in_segment
        img.put_u32(starts + 8, seg_info_rel as u32); // __DATA: fixups here

        // dyld_chained_starts_in_segment for __DATA.
        let seg = starts + seg_info_rel;
        img.put_u32(seg, 24); // size
        img.put_u16(seg + 4, 0x4000); // page_size (16 KiB)
        img.put_u16(seg + 6, pointer_format); // pointer_format
        img.put_u64(seg + 8, DATA_FILEOFF); // segment_offset (file offset of __DATA)
        img.put_u32(seg + 16, 0); // max_valid_pointer
        img.put_u16(seg + 20, 1); // page_count
        img.put_u16(seg + 22, 0); // page_start[0] = first fixup at page offset 0

        // Imports array (per `imports_format`) + symbol pool.
        let mut pool = symbols_off_rel; // relative name offset cursor
        let mut expected = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let mangled = format!("_{name}");
            let name_off = (pool - symbols_off_rel) as u32;
            put_chained_import(
                &mut img,
                blob_off + imports_off_rel,
                i,
                imports_format,
                name_off,
                1,
            );
            img.put_str(blob_off + pool, &mangled);
            pool += mangled.len() + 1;
            expected.push((
                normalize_symbol_name(&mangled).unwrap(),
                DATA_VMADDR + (i * 8) as u64,
            ));
        }

        // The __DATA chain: one non-authenticated bind slot per import, linked by
        // `next`. arm64e callers that want authenticated slots rewrite them.
        for i in 0..n {
            let slot = data_off + i * 8;
            let is_last = i + 1 == n;
            img.put_u64(
                slot,
                encode_chain_bind(pointer_format, i as u64, false, is_last),
            );
        }

        (img.bytes, expected)
    }

    #[test]
    fn chained_fixups_64_offset_binds() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let (bytes, expected) = build_chained(
            &layout,
            DYLD_CHAINED_PTR_64_OFFSET,
            DYLD_CHAINED_IMPORT,
            &["open", "fstat$INODE64", "close"],
        );
        let view = FileView::new(&bytes, 0).expect("valid chained-fixups image");
        let imports = parse_imports(&view).expect("enumerate chained-fixups imports");

        for (name, slot) in &expected {
            let import = find(&imports, name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(import.slot as u64, *slot, "chained slot vmaddr for {name}");
            assert!(
                !import.authenticated,
                "64_offset binds are not authenticated"
            );
            assert_eq!(&*import.library, "libSystem.B.dylib");
        }
        // `fstat$INODE64` normalized to `fstat`.
        assert!(find(&imports, "fstat").is_some());
        assert_eq!(imports.len(), 3);
    }

    #[test]
    fn arm64e_authenticated_binds_are_flagged() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: arch::CPU_SUBTYPE_ARM64E,
        };
        // Build with the arm64e userland format, then set each slot's per-slot auth
        // bit (build_chained emits non-authenticated binds by default).
        let (mut bytes, expected) = build_chained(
            &layout,
            DYLD_CHAINED_PTR_ARM64E_USERLAND,
            DYLD_CHAINED_IMPORT,
            &["open", "read"],
        );
        let n = expected.len();
        for i in 0..n {
            let slot = DATA_FILEOFF as usize + i * 8;
            let raw =
                encode_chain_bind(DYLD_CHAINED_PTR_ARM64E_USERLAND, i as u64, true, i + 1 == n);
            bytes[slot..slot + 8].copy_from_slice(&raw.to_le_bytes());
        }
        let view = FileView::new(&bytes, 0).expect("valid arm64e image");
        let imports = parse_imports(&view).expect("enumerate arm64e imports");
        assert_eq!(imports.len(), n);
        for import in &imports {
            assert!(
                import.authenticated,
                "arm64e auth bind must be flagged for refusal"
            );
        }
    }

    #[test]
    fn legacy_dyld_info_binds() {
        // A minimal image with an LC_DYLD_INFO_ONLY bind stream binding two symbols
        // into __DATA via SET_SEGMENT_AND_OFFSET + DO_BIND.
        let data_seg_index = 1u8; // __DATA is the 2nd segment (index 1)
        let blob_off = 0x5000usize;
        let mut img = Image::new(0x8000);
        img.put_u32(0, MH_MAGIC_64);
        img.put_u32(4, arch::CPU_TYPE_X86_64);
        img.put_u32(8, 0);
        img.put_u32(12, 0x6);

        let mut c = 32usize;
        let mut ncmds = 0u32;

        let text = c;
        img.put_u32(text, LC_SEGMENT_64);
        img.put_u32(text + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(text + 8, "__TEXT");
        img.put_u64(text + 24, TEXT_VMADDR);
        img.put_u64(text + 32, 0x4000);
        img.put_u64(text + 40, 0);
        img.put_u64(text + 48, 0x4000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        let data = c;
        img.put_u32(data, LC_SEGMENT_64);
        img.put_u32(data + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(data + 8, "__DATA");
        img.put_u64(data + 24, DATA_VMADDR);
        img.put_u64(data + 32, 0x1000);
        img.put_u64(data + 40, DATA_FILEOFF);
        img.put_u64(data + 48, 0x1000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        let link = c;
        img.put_u32(link, LC_SEGMENT_64);
        img.put_u32(link + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(link + 8, "__LINKEDIT");
        img.put_u64(link + 24, DATA_VMADDR + 0x1000);
        img.put_u64(link + 32, 0x3000);
        img.put_u64(link + 40, blob_off as u64);
        img.put_u64(link + 48, 0x3000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        let dylib = c;
        let dylib_size = (24 + "libSystem.B.dylib".len() + 1).next_multiple_of(8);
        img.put_u32(dylib, LC_LOAD_DYLIB);
        img.put_u32(dylib + 4, dylib_size as u32);
        img.put_u32(dylib + 8, 24);
        img.put_str(dylib + 24, "libSystem.B.dylib");
        c += dylib_size;
        ncmds += 1;

        // LC_DYLD_INFO_ONLY with a bind stream.
        let info = c;
        img.put_u32(info, LC_DYLD_INFO_ONLY);
        img.put_u32(info + 4, 48);
        // Build the bind opcode stream in __LINKEDIT.
        let stream = blob_off;
        let mut s = stream;
        // SET_DYLIB_ORDINAL_IMM 1
        img.bytes[s] = BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | 1;
        s += 1;
        // SET_SYMBOL_TRAILING_FLAGS_IMM "_open"
        img.bytes[s] = BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM;
        s += 1;
        img.put_str(s, "_open");
        s += "_open".len() + 1;
        // SET_TYPE_IMM 1
        img.bytes[s] = BIND_OPCODE_SET_TYPE_IMM | 1;
        s += 1;
        // SET_SEGMENT_AND_OFFSET_ULEB seg=1 off=0
        img.bytes[s] = BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | data_seg_index;
        s += 1;
        img.bytes[s] = 0; // uleb 0
        s += 1;
        // DO_BIND (binds _open at __DATA+0)
        img.bytes[s] = BIND_OPCODE_DO_BIND;
        s += 1;
        // SET_SYMBOL_TRAILING_FLAGS_IMM "_close"
        img.bytes[s] = BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM;
        s += 1;
        img.put_str(s, "_close");
        s += "_close".len() + 1;
        // DO_BIND (seg_offset auto-advanced by 8 → __DATA+8)
        img.bytes[s] = BIND_OPCODE_DO_BIND;
        s += 1;
        img.bytes[s] = BIND_OPCODE_DONE;
        s += 1;
        let bind_size = s - stream;
        img.put_u32(info + 16, stream as u32); // bind_off
        img.put_u32(info + 20, bind_size as u32); // bind_size
        c += 48;
        ncmds += 1;

        img.put_u32(16, ncmds);
        img.put_u32(20, (c - 32) as u32);

        let view = FileView::new(&img.bytes, 0).expect("valid dyld-info image");
        let imports = parse_imports(&view).expect("enumerate legacy binds");
        let open = find(&imports, "open").expect("open bind");
        assert_eq!(open.slot as u64, DATA_VMADDR, "first bind at __DATA+0");
        let close = find(&imports, "close").expect("close bind");
        assert_eq!(
            close.slot as u64,
            DATA_VMADDR + 8,
            "second bind auto-advanced +8"
        );
        assert_eq!(&*open.library, "libSystem.B.dylib");
    }

    #[test]
    fn rejects_32bit_and_fat_and_foreign_endian() {
        // 32-bit thin.
        let mut b = vec![0u8; 64];
        b[0..4].copy_from_slice(&MH_MAGIC_32.to_le_bytes());
        assert!(matches!(classify(&b), Err(Error::Unsupported(_))));
        // Byte-swapped 64-bit.
        b[0..4].copy_from_slice(&MH_CIGAM_64.to_le_bytes());
        assert!(matches!(classify(&b), Err(Error::Unsupported(_))));
        // Not a Mach-O at all.
        b[0..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        assert!(matches!(classify(&b), Err(Error::Malformed(_))));
    }

    #[test]
    fn unsupported_cputype_is_rejected() {
        let layout = Layout {
            cpu_type: 0x0000_0007,
            cpu_subtype: 0,
        }; // CPU_TYPE_X86 (32-bit)
        let (bytes, _) = build_indirect(&layout, &["open"]);
        let view = FileView::new(&bytes, 0).unwrap();
        assert!(matches!(parse_imports(&view), Err(Error::Unsupported(_))));
    }

    /// Build a synthetic thin image whose `__DATA` slots are enumerable by BOTH the
    /// indirect symbol table (a `__got` / `S_NON_LAZY_SYMBOL_POINTERS` section) AND a
    /// chained-fixups blob threading the same slots. Each `slots` entry is
    /// `(symbol, authenticated)`: the chain marks that slot's precise arm64e auth bit,
    /// while the indirect table (which cannot see it) carries only the coarse
    /// whole-image flag. Exercises the dedup/merge (M3) and the auth override (m2).
    /// Returns `(bytes, [(name, slot_vmaddr)])`.
    fn build_indirect_and_chained(
        layout: &Layout,
        pointer_format: u16,
        slots: &[(&str, bool)],
    ) -> (Vec<u8>, Vec<(String, u64)>) {
        let n = slots.len();
        let data_off = DATA_FILEOFF as usize;
        let blob_off = 0x5000usize;
        let indirect_off = 0x6000usize;
        let symtab_off = 0x6100usize;
        let strtab_off = 0x6200usize;
        let strtab_size = 0x8000 - strtab_off;
        let mut img = Image::new(0x8000);

        img.put_u32(0, MH_MAGIC_64);
        img.put_u32(4, layout.cpu_type);
        img.put_u32(8, layout.cpu_subtype);
        img.put_u32(12, 0x6); // MH_DYLIB

        let mut c = 32usize;
        let mut ncmds = 0u32;

        // __TEXT.
        let text = c;
        img.put_u32(text, LC_SEGMENT_64);
        img.put_u32(text + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(text + 8, "__TEXT");
        img.put_u64(text + 24, TEXT_VMADDR);
        img.put_u64(text + 32, 0x4000);
        img.put_u64(text + 40, 0);
        img.put_u64(text + 48, 0x4000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // __DATA with one non-lazy symbol-pointer section (`__got`) over the slots.
        let data = c;
        let seg_and_one_sec = SEGMENT_COMMAND_64_SIZE + SECTION_64_SIZE;
        img.put_u32(data, LC_SEGMENT_64);
        img.put_u32(data + 4, seg_and_one_sec as u32);
        img.put_str(data + 8, "__DATA");
        img.put_u64(data + 24, DATA_VMADDR);
        img.put_u64(data + 32, 0x1000);
        img.put_u64(data + 40, DATA_FILEOFF);
        img.put_u64(data + 48, 0x1000);
        img.put_u32(data + 64, 1); // nsects
        let sec = data + SEGMENT_COMMAND_64_SIZE as usize;
        img.put_str(sec, "__got");
        img.put_str(sec + 16, "__DATA");
        img.put_u64(sec + 32, DATA_VMADDR); // addr
        img.put_u64(sec + 40, (n * 8) as u64); // size
        img.put_u32(sec + 48, data_off as u32); // offset
        img.put_u32(sec + 64, S_NON_LAZY_SYMBOL_POINTERS); // flags
        img.put_u32(sec + 68, 0); // reserved1 = indirect base index 0
        c += seg_and_one_sec as usize;
        ncmds += 1;

        // __LINKEDIT (chained blob + indirect/symtab/strtab).
        let link = c;
        img.put_u32(link, LC_SEGMENT_64);
        img.put_u32(link + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(link + 8, "__LINKEDIT");
        img.put_u64(link + 24, DATA_VMADDR + 0x1000);
        img.put_u64(link + 32, 0x3000);
        img.put_u64(link + 40, blob_off as u64);
        img.put_u64(link + 48, 0x3000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // LC_LOAD_DYLIB (ordinal 1).
        let dylib = c;
        let dylib_size = (24 + "libSystem.B.dylib".len() + 1).next_multiple_of(8);
        img.put_u32(dylib, LC_LOAD_DYLIB);
        img.put_u32(dylib + 4, dylib_size as u32);
        img.put_u32(dylib + 8, 24);
        img.put_str(dylib + 24, "libSystem.B.dylib");
        c += dylib_size;
        ncmds += 1;

        // LC_SYMTAB.
        let symtab = c;
        img.put_u32(symtab, LC_SYMTAB);
        img.put_u32(symtab + 4, 24);
        img.put_u32(symtab + 8, symtab_off as u32);
        img.put_u32(symtab + 12, n as u32);
        img.put_u32(symtab + 16, strtab_off as u32);
        img.put_u32(symtab + 20, strtab_size as u32);
        c += 24;
        ncmds += 1;

        // LC_DYSYMTAB.
        let dysym = c;
        img.put_u32(dysym, LC_DYSYMTAB);
        img.put_u32(dysym + 4, 80);
        img.put_u32(dysym + 56, indirect_off as u32);
        img.put_u32(dysym + 60, n as u32);
        c += 80;
        ncmds += 1;

        // LC_DYLD_CHAINED_FIXUPS.
        let cf = c;
        img.put_u32(cf, LC_DYLD_CHAINED_FIXUPS);
        img.put_u32(cf + 4, 16);
        img.put_u32(cf + 8, blob_off as u32);
        img.put_u32(cf + 12, 0x1000);
        c += 16;
        ncmds += 1;

        img.put_u32(16, ncmds);
        img.put_u32(20, (c - 32) as u32);

        // Indirect table + symbol table + string table (the fishhook path).
        let mut strx = 1u32; // index 0 is the leading NUL
        let mut expected = Vec::new();
        for (i, (name, _auth)) in slots.iter().enumerate() {
            let mangled = format!("_{name}");
            img.put_u32(indirect_off + i * 4, i as u32); // indirect[i] = symbol i
            let nl = symtab_off + i * 16;
            img.put_u32(nl, strx); // n_strx
            img.bytes[nl + 4] = N_UNDF | N_EXT; // undefined external
            img.put_u16(nl + 6, 1 << 8); // library ordinal 1
            img.put_str(strtab_off + strx as usize, &mangled);
            strx += mangled.len() as u32 + 1;
            expected.push((
                normalize_symbol_name(&mangled).unwrap(),
                DATA_VMADDR + (i * 8) as u64,
            ));
        }

        // Chained-fixups blob threading the SAME __DATA slots (the precise-auth path).
        let starts_off_rel = 0x40usize;
        let imports_off_rel = 0x100usize;
        let symbols_off_rel = 0x200usize;
        img.put_u32(blob_off, 0); // fixups_version
        img.put_u32(blob_off + 4, starts_off_rel as u32);
        img.put_u32(blob_off + 8, imports_off_rel as u32);
        img.put_u32(blob_off + 12, symbols_off_rel as u32);
        img.put_u32(blob_off + 16, n as u32); // imports_count
        img.put_u32(blob_off + 20, DYLD_CHAINED_IMPORT); // imports_format

        let starts = blob_off + starts_off_rel;
        img.put_u32(starts, 2); // seg_count (0=__TEXT, 1=__DATA)
        img.put_u32(starts + 4, 0); // __TEXT: no fixups
        let seg_info_rel = 0x20usize;
        img.put_u32(starts + 8, seg_info_rel as u32); // __DATA: fixups here

        let seg = starts + seg_info_rel;
        img.put_u32(seg, 24); // size
        img.put_u16(seg + 4, 0x4000); // page_size
        img.put_u16(seg + 6, pointer_format);
        img.put_u64(seg + 8, DATA_FILEOFF); // segment_offset
        img.put_u32(seg + 16, 0); // max_valid_pointer
        img.put_u16(seg + 20, 1); // page_count
        img.put_u16(seg + 22, 0); // page_start[0] = 0

        let mut pool = symbols_off_rel;
        for (i, (name, _auth)) in slots.iter().enumerate() {
            let mangled = format!("_{name}");
            let name_off = (pool - symbols_off_rel) as u32;
            put_chained_import(
                &mut img,
                blob_off + imports_off_rel,
                i,
                DYLD_CHAINED_IMPORT,
                name_off,
                1,
            );
            img.put_str(blob_off + pool, &mangled);
            pool += mangled.len() + 1;
        }
        for (i, (_name, auth)) in slots.iter().enumerate() {
            let slot = data_off + i * 8;
            img.put_u64(
                slot,
                encode_chain_bind(pointer_format, i as u64, *auth, i + 1 == n),
            );
        }

        (img.bytes, expected)
    }

    /// Build a chained-fixups image with MULTIPLE data pages. `pages[j]` lists the
    /// bind symbols on page `j`; an empty page emits `DYLD_CHAINED_PTR_START_NONE`.
    /// Ordinals are assigned in flat page-then-slot order. Returns
    /// `(bytes, [(name, slot_vmaddr)])`.
    fn build_chained_multipage(
        layout: &Layout,
        pointer_format: u16,
        pages: &[&[&str]],
        page_size: u64,
    ) -> (Vec<u8>, Vec<(String, u64)>) {
        let data_off = DATA_FILEOFF as usize;
        let blob_off = 0x5000usize;
        let mut img = Image::new(0x8000);

        img.put_u32(0, MH_MAGIC_64);
        img.put_u32(4, layout.cpu_type);
        img.put_u32(8, layout.cpu_subtype);
        img.put_u32(12, 0x6);

        let mut c = 32usize;
        let mut ncmds = 0u32;

        // __TEXT.
        let text = c;
        img.put_u32(text, LC_SEGMENT_64);
        img.put_u32(text + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(text + 8, "__TEXT");
        img.put_u64(text + 24, TEXT_VMADDR);
        img.put_u64(text + 32, 0x4000);
        img.put_u64(text + 40, 0);
        img.put_u64(text + 48, 0x4000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // __DATA.
        let data = c;
        img.put_u32(data, LC_SEGMENT_64);
        img.put_u32(data + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(data + 8, "__DATA");
        img.put_u64(data + 24, DATA_VMADDR);
        img.put_u64(data + 32, 0x1000);
        img.put_u64(data + 40, DATA_FILEOFF);
        img.put_u64(data + 48, 0x1000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // __LINKEDIT.
        let link = c;
        img.put_u32(link, LC_SEGMENT_64);
        img.put_u32(link + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(link + 8, "__LINKEDIT");
        img.put_u64(link + 24, DATA_VMADDR + 0x1000);
        img.put_u64(link + 32, 0x3000);
        img.put_u64(link + 40, blob_off as u64);
        img.put_u64(link + 48, 0x3000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        // LC_LOAD_DYLIB.
        let dylib = c;
        let dylib_size = (24 + "libSystem.B.dylib".len() + 1).next_multiple_of(8);
        img.put_u32(dylib, LC_LOAD_DYLIB);
        img.put_u32(dylib + 4, dylib_size as u32);
        img.put_u32(dylib + 8, 24);
        img.put_str(dylib + 24, "libSystem.B.dylib");
        c += dylib_size;
        ncmds += 1;

        // LC_DYLD_CHAINED_FIXUPS.
        let cf = c;
        img.put_u32(cf, LC_DYLD_CHAINED_FIXUPS);
        img.put_u32(cf + 4, 16);
        img.put_u32(cf + 8, blob_off as u32);
        img.put_u32(cf + 12, 0x1000);
        c += 16;
        ncmds += 1;

        img.put_u32(16, ncmds);
        img.put_u32(20, (c - 32) as u32);

        // Flat import list across all pages (ordinal = flat index).
        let flat: Vec<&str> = pages.iter().flat_map(|p| p.iter().copied()).collect();
        let n = flat.len();

        let starts_off_rel = 0x40usize;
        let imports_off_rel = 0x100usize;
        let symbols_off_rel = 0x200usize;
        img.put_u32(blob_off, 0);
        img.put_u32(blob_off + 4, starts_off_rel as u32);
        img.put_u32(blob_off + 8, imports_off_rel as u32);
        img.put_u32(blob_off + 12, symbols_off_rel as u32);
        img.put_u32(blob_off + 16, n as u32);
        img.put_u32(blob_off + 20, DYLD_CHAINED_IMPORT);

        let starts = blob_off + starts_off_rel;
        img.put_u32(starts, 2); // seg_count (0=__TEXT, 1=__DATA)
        img.put_u32(starts + 4, 0);
        let seg_info_rel = 0x20usize;
        img.put_u32(starts + 8, seg_info_rel as u32);

        // dyld_chained_starts_in_segment with `pages.len()` pages.
        let seg = starts + seg_info_rel;
        let page_count = pages.len();
        img.put_u32(seg, 22 + page_count as u32 * 2); // size
        img.put_u16(seg + 4, page_size as u16);
        img.put_u16(seg + 6, pointer_format);
        img.put_u64(seg + 8, DATA_FILEOFF); // segment_offset
        img.put_u32(seg + 16, 0);
        img.put_u16(seg + 20, page_count as u16); // page_count
        for (j, page) in pages.iter().enumerate() {
            let start = if page.is_empty() {
                DYLD_CHAINED_PTR_START_NONE
            } else {
                0
            };
            img.put_u16(seg + 22 + j * 2, start);
        }

        // Imports array + symbol pool.
        let mut pool = symbols_off_rel;
        for (i, name) in flat.iter().enumerate() {
            let mangled = format!("_{name}");
            let name_off = (pool - symbols_off_rel) as u32;
            put_chained_import(
                &mut img,
                blob_off + imports_off_rel,
                i,
                DYLD_CHAINED_IMPORT,
                name_off,
                1,
            );
            img.put_str(blob_off + pool, &mangled);
            pool += mangled.len() + 1;
        }

        // Per-page chains: slots 8 bytes apart; `next = 0` ends each page's chain.
        let mut ordinal = 0usize;
        let mut expected = Vec::new();
        for (j, page) in pages.iter().enumerate() {
            for (k, name) in page.iter().enumerate() {
                let file = data_off + j * page_size as usize + k * 8;
                img.put_u64(
                    file,
                    encode_chain_bind(pointer_format, ordinal as u64, false, k + 1 == page.len()),
                );
                let vmaddr = DATA_VMADDR + (j * page_size as usize + k * 8) as u64;
                expected.push((normalize_symbol_name(&format!("_{name}")).unwrap(), vmaddr));
                ordinal += 1;
            }
        }

        (img.bytes, expected)
    }

    /// An arm64e image enumerated ONLY via the indirect symbol table: the path cannot
    /// see a per-slot auth bit, so every slot is coarsely flagged authenticated for
    /// refusal (R1) — previously untested.
    #[test]
    fn indirect_arm64e_flags_all_slots_authenticated() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: arch::CPU_SUBTYPE_ARM64E,
        };
        let (bytes, expected) = build_indirect(&layout, &["open", "read", "close"]);
        let view = FileView::new(&bytes, 0).expect("valid arm64e indirect image");
        assert!(view.is_arm64e());
        let imports = parse_imports(&view).expect("enumerate indirect imports");
        assert_eq!(imports.len(), expected.len());
        for import in &imports {
            assert!(
                import.authenticated,
                "arm64e indirect slot is coarsely flagged authenticated"
            );
        }
    }

    /// `DYLD_CHAINED_PTR_64` (format 2, non-offset) decodes the bind fields identically
    /// to `_64_OFFSET`; only the (unused) target semantics differ.
    #[test]
    fn chained_ptr_64_non_offset_binds() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let (bytes, expected) = build_chained(
            &layout,
            DYLD_CHAINED_PTR_64,
            DYLD_CHAINED_IMPORT,
            &["open", "read", "close"],
        );
        let view = FileView::new(&bytes, 0).expect("valid PTR_64 image");
        let imports = parse_imports(&view).expect("enumerate PTR_64 binds");
        for (name, slot) in &expected {
            let import = find(&imports, name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(import.slot as u64, *slot);
            assert!(!import.authenticated);
        }
        assert_eq!(imports.len(), 3);
    }

    /// The 24-bit-ordinal arm64e userland format; a non-authenticated bind here is
    /// rebindable.
    #[test]
    fn chained_arm64e_userland24_binds() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: arch::CPU_SUBTYPE_ARM64E,
        };
        let (bytes, expected) = build_chained(
            &layout,
            DYLD_CHAINED_PTR_ARM64E_USERLAND24,
            DYLD_CHAINED_IMPORT,
            &["open", "read"],
        );
        let view = FileView::new(&bytes, 0).expect("valid arm64e userland24 image");
        let imports = parse_imports(&view).expect("enumerate userland24 binds");
        for (name, slot) in &expected {
            let import = find(&imports, name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(import.slot as u64, *slot);
            assert!(
                !import.authenticated,
                "non-auth userland24 bind is rebindable"
            );
        }
        assert_eq!(imports.len(), 2);
    }

    /// The `DYLD_CHAINED_IMPORT_ADDEND` (8-byte) and `_ADDEND64` (16-byte) import
    /// record formats resolve names/libraries correctly (the parser strides over the
    /// addend field).
    #[test]
    fn chained_import_addend_and_addend64_formats() {
        for imports_format in [DYLD_CHAINED_IMPORT_ADDEND, DYLD_CHAINED_IMPORT_ADDEND64] {
            let layout = Layout {
                cpu_type: arch::CPU_TYPE_X86_64,
                cpu_subtype: 0,
            };
            let (bytes, expected) = build_chained(
                &layout,
                DYLD_CHAINED_PTR_64_OFFSET,
                imports_format,
                &["open", "stat"],
            );
            let view = FileView::new(&bytes, 0).expect("valid addend-format image");
            let imports = parse_imports(&view)
                .unwrap_or_else(|e| panic!("enumerate imports_format {imports_format}: {e:?}"));
            for (name, slot) in &expected {
                let import = find(&imports, name)
                    .unwrap_or_else(|| panic!("missing {name} ({imports_format})"));
                assert_eq!(import.slot as u64, *slot);
                assert_eq!(&*import.library, "libSystem.B.dylib");
            }
            assert_eq!(imports.len(), 2, "imports_format {imports_format}");
        }
    }

    /// Multi-page chains: page 0 and page 2 hold bind chains, page 1 is
    /// `DYLD_CHAINED_PTR_START_NONE` and is skipped. Every bind across the live pages
    /// is recovered at its own page-relative slot address.
    #[test]
    fn chained_multipage_skips_start_none() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let page_size = 0x40u64;
        let (bytes, expected) = build_chained_multipage(
            &layout,
            DYLD_CHAINED_PTR_64_OFFSET,
            &[&["open", "read"], &[], &["close", "stat"]],
            page_size,
        );
        let view = FileView::new(&bytes, 0).expect("valid multi-page image");
        let imports = parse_imports(&view).expect("enumerate multi-page binds");
        assert_eq!(imports.len(), 4);
        for (name, slot) in &expected {
            let import = find(&imports, name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(import.slot as u64, *slot, "multi-page slot for {name}");
        }
        // Page 2's first slot sits a full page beyond page 0's.
        let close = find(&imports, "close").expect("close").slot as u64;
        assert_eq!(
            close,
            DATA_VMADDR + 2 * page_size,
            "page-2 chain starts one page in"
        );
    }

    /// Indirect entries flagged `INDIRECT_SYMBOL_LOCAL` / `INDIRECT_SYMBOL_ABS` carry
    /// no import name and must be skipped, leaving only the real undefined-external.
    #[test]
    fn indirect_symbol_local_and_abs_are_skipped() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let (mut bytes, expected) = build_indirect(&layout, &["open", "local", "abs"]);
        // build_indirect lays the indirect table at 0x5000, one u32 per slot.
        let indirect_off = 0x5000usize;
        bytes[indirect_off + 4..indirect_off + 8]
            .copy_from_slice(&INDIRECT_SYMBOL_LOCAL.to_le_bytes());
        bytes[indirect_off + 8..indirect_off + 12]
            .copy_from_slice(&INDIRECT_SYMBOL_ABS.to_le_bytes());
        let view = FileView::new(&bytes, 0).expect("valid indirect image");
        let imports = parse_imports(&view).expect("enumerate with sentinels");
        assert_eq!(imports.len(), 1, "LOCAL and ABS slots are skipped");
        assert_eq!(
            find(&imports, "open").expect("open").slot as u64,
            expected[0].1
        );
        assert!(find(&imports, "local").is_none());
        assert!(find(&imports, "abs").is_none());
    }

    /// The dedup/merge with BOTH sources present: an arm64e `__got` enumerated by the
    /// indirect table (coarse auth=true) and the chain (precise per-slot auth). After
    /// the merge each slot's auth is the CHAIN's precise bit — the non-authenticated
    /// slot is rebindable (m2) and the authenticated one stays refused (R1) — and the
    /// two sources dedup to one entry per slot (M3/m2).
    #[test]
    fn indirect_and_chained_merge_prefers_chained_auth() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: arch::CPU_SUBTYPE_ARM64E,
        };
        let (bytes, expected) = build_indirect_and_chained(
            &layout,
            DYLD_CHAINED_PTR_ARM64E_USERLAND,
            &[("open", false), ("read", true)],
        );
        let view = FileView::new(&bytes, 0).expect("valid combined image");
        let imports = parse_imports(&view).expect("enumerate combined image");
        assert_eq!(
            imports.len(),
            2,
            "indirect + chained slots dedup to one each"
        );
        let open = find(&imports, "open").expect("open");
        assert_eq!(open.slot as u64, expected[0].1);
        assert!(
            !open.authenticated,
            "non-auth __got slot on arm64e is rebindable (m2)"
        );
        let read = find(&imports, "read").expect("read");
        assert_eq!(read.slot as u64, expected[1].1);
        assert!(read.authenticated, "authenticated slot stays refused (R1)");
    }

    /// The chained-fixups complement is best-effort: an unparseable secondary blob
    /// must NOT discard the primary indirect enumeration (M3). Here the primary
    /// indirect table is intact but the chained blob's header is corrupted, so
    /// `parse_chained_fixups` fails — the indirect imports must still be returned.
    #[test]
    fn chained_complement_failure_keeps_indirect() {
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let (mut bytes, expected) = build_indirect_and_chained(
            &layout,
            DYLD_CHAINED_PTR_64_OFFSET,
            &[("open", false), ("read", false)],
        );
        // Corrupt fixups_version (blob at 0x5000) so parse_chained_fixups → Unsupported.
        let blob_off = 0x5000usize;
        bytes[blob_off..blob_off + 4].copy_from_slice(&99u32.to_le_bytes());
        let view = FileView::new(&bytes, 0).expect("valid indirect, broken chain");
        let imports = parse_imports(&view).expect("indirect survives a broken complement");
        assert_eq!(imports.len(), 2, "the two indirect slots are retained");
        for (name, slot) in &expected {
            assert_eq!(
                find(&imports, name).expect("indirect import").slot as u64,
                *slot
            );
        }
    }

    /// The arm64e chain stride is per-format: 4 bytes for KERNEL/FIRMWARE, 8 bytes for
    /// the userland/desktop formats; the 64-bit formats use a 12-bit next scaled by 4
    /// (Apple `<mach-o/fixup-chains.h>`).
    #[test]
    fn decode_chain_stride_is_per_format() {
        // arm64e bind, auth=0, next-field = 3.
        let raw = (1u64 << 62) | (3u64 << 51);
        for pf in [
            DYLD_CHAINED_PTR_ARM64E_KERNEL,
            DYLD_CHAINED_PTR_ARM64E_FIRMWARE,
        ] {
            assert_eq!(
                decode_chain(raw, pf).expect("kernel/firmware decodes").next,
                3 * 4
            );
        }
        for pf in [
            DYLD_CHAINED_PTR_ARM64E,
            DYLD_CHAINED_PTR_ARM64E_USERLAND,
            DYLD_CHAINED_PTR_ARM64E_USERLAND24,
        ] {
            assert_eq!(decode_chain(raw, pf).expect("userland decodes").next, 3 * 8);
        }
        // 64-bit format: bind at bit 63, 12-bit next scaled by 4.
        let raw64 = (1u64 << 63) | (3u64 << 51);
        assert_eq!(
            decode_chain(raw64, DYLD_CHAINED_PTR_64)
                .expect("ptr64 decodes")
                .next,
            3 * 4
        );
    }

    /// Build a thin image with an `LC_DYLD_INFO_ONLY` bind stream `stream` placed in
    /// `__LINKEDIT` (segment index 1 is `__DATA`). Returns the bytes.
    fn build_legacy_bind(layout: &Layout, stream: &[u8]) -> Vec<u8> {
        let blob_off = 0x5000usize;
        let mut img = Image::new(0x8000);
        img.put_u32(0, MH_MAGIC_64);
        img.put_u32(4, layout.cpu_type);
        img.put_u32(8, layout.cpu_subtype);
        img.put_u32(12, 0x6);

        let mut c = 32usize;
        let mut ncmds = 0u32;

        let text = c;
        img.put_u32(text, LC_SEGMENT_64);
        img.put_u32(text + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(text + 8, "__TEXT");
        img.put_u64(text + 24, TEXT_VMADDR);
        img.put_u64(text + 32, 0x4000);
        img.put_u64(text + 40, 0);
        img.put_u64(text + 48, 0x4000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        let data = c;
        img.put_u32(data, LC_SEGMENT_64);
        img.put_u32(data + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(data + 8, "__DATA");
        img.put_u64(data + 24, DATA_VMADDR);
        img.put_u64(data + 32, 0x1000);
        img.put_u64(data + 40, DATA_FILEOFF);
        img.put_u64(data + 48, 0x1000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        let link = c;
        img.put_u32(link, LC_SEGMENT_64);
        img.put_u32(link + 4, SEGMENT_COMMAND_64_SIZE as u32);
        img.put_str(link + 8, "__LINKEDIT");
        img.put_u64(link + 24, DATA_VMADDR + 0x1000);
        img.put_u64(link + 32, 0x3000);
        img.put_u64(link + 40, blob_off as u64);
        img.put_u64(link + 48, 0x3000);
        c += SEGMENT_COMMAND_64_SIZE as usize;
        ncmds += 1;

        let dylib = c;
        let dylib_size = (24 + "libSystem.B.dylib".len() + 1).next_multiple_of(8);
        img.put_u32(dylib, LC_LOAD_DYLIB);
        img.put_u32(dylib + 4, dylib_size as u32);
        img.put_u32(dylib + 8, 24);
        img.put_str(dylib + 24, "libSystem.B.dylib");
        c += dylib_size;
        ncmds += 1;

        let info = c;
        img.put_u32(info, LC_DYLD_INFO_ONLY);
        img.put_u32(info + 4, 48);
        img.bytes[blob_off..blob_off + stream.len()].copy_from_slice(stream);
        img.put_u32(info + 16, blob_off as u32); // bind_off
        img.put_u32(info + 20, stream.len() as u32); // bind_size
        c += 48;
        ncmds += 1;

        img.put_u32(16, ncmds);
        img.put_u32(20, (c - 32) as u32);
        img.bytes
    }

    /// A non-UTF-8 symbol name in a legacy bind stream must advance the opcode cursor
    /// by its RAW byte length, not the UTF-8-lossy `String` length (which inflates by
    /// 2 bytes per invalid byte). Otherwise the cursor over-advances and desyncs the
    /// following opcodes (m3): here the clean second bind must still land at __DATA+8.
    #[test]
    fn non_utf8_bind_symbol_name_keeps_cursor_in_sync() {
        let mut s: Vec<u8> = Vec::new();
        s.push(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | 1);
        // A name with an invalid-UTF-8 byte: 6 raw bytes, but an 8-byte lossy String.
        s.push(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM);
        s.extend_from_slice(b"_ope\xffn");
        s.push(0);
        s.push(BIND_OPCODE_SET_TYPE_IMM | 1);
        s.push(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | 1); // seg 1 (__DATA), offset:
        s.push(0); // uleb 0
        s.push(BIND_OPCODE_DO_BIND); // bind #1 at __DATA+0, then advance +8
        s.push(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM);
        s.extend_from_slice(b"_close");
        s.push(0);
        s.push(BIND_OPCODE_DO_BIND); // bind #2 at __DATA+8
        s.push(BIND_OPCODE_DONE);

        let layout = Layout {
            cpu_type: arch::CPU_TYPE_X86_64,
            cpu_subtype: 0,
        };
        let bytes = build_legacy_bind(&layout, &s);
        let view = FileView::new(&bytes, 0).expect("valid legacy-bind image");
        let imports = parse_imports(&view).expect("enumerate binds past a non-UTF-8 name");
        assert_eq!(
            imports.len(),
            2,
            "both binds recovered — cursor stayed in sync"
        );
        let close = find(&imports, "close").expect("clean second bind resolved");
        assert_eq!(
            close.slot as u64,
            DATA_VMADDR + 8,
            "second bind at __DATA+8 (no desync)"
        );
    }

    #[test]
    fn malformed_images_never_read_out_of_bounds() {
        // A dependency-free xorshift PRNG keeps the fuzz self-contained.
        let mut state: u64 = 0x1234_5678_9abc_def1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // Wholly random buffers: every outcome is a typed Result, never a panic.
        for _ in 0..8000 {
            let len = (next() as usize % 4096) + 4;
            let mut buf = vec![0u8; len];
            for b in &mut buf {
                *b = next() as u8;
            }
            if let Ok(Container::Thin) = classify(&buf)
                && let Ok(view) = FileView::new(&buf, 0)
            {
                let _ = parse_imports(&view);
            }
        }

        // Corrupt valid images across their whole length (header, load commands, the
        // linkedit tables, and the __DATA chain), proving the bounds checks hold no
        // matter which field a flip lands in. Restored in place each iteration.
        let mut fuzz = |base: &mut Vec<u8>| {
            let len = base.len();
            for _ in 0..30_000 {
                let idx = next() as usize % len;
                let orig = base[idx];
                base[idx] ^= (next() as u8) | 1;
                if let Ok(view) = FileView::new(base, 0) {
                    let _ = parse_imports(&view);
                }
                base[idx] = orig;
            }
        };
        let layout = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: 0,
        };
        let (mut indirect, _) = build_indirect(&layout, &["open", "close", "stat"]);
        fuzz(&mut indirect);
        let (mut chained, _) = build_chained(
            &layout,
            DYLD_CHAINED_PTR_64_OFFSET,
            DYLD_CHAINED_IMPORT,
            &["open", "read", "write"],
        );
        fuzz(&mut chained);
        // Also fuzz an arm64e chained image (arm64e decode + ADDEND imports) and a
        // combined indirect+chained image (the dedup/merge path), so every source and
        // the merge are exercised against corruption.
        let arm64e = Layout {
            cpu_type: arch::CPU_TYPE_ARM64,
            cpu_subtype: arch::CPU_SUBTYPE_ARM64E,
        };
        let (mut chained_arm64e, _) = build_chained(
            &arm64e,
            DYLD_CHAINED_PTR_ARM64E_USERLAND,
            DYLD_CHAINED_IMPORT_ADDEND,
            &["open", "read"],
        );
        fuzz(&mut chained_arm64e);
        let (mut combined, _) = build_indirect_and_chained(
            &arm64e,
            DYLD_CHAINED_PTR_ARM64E_USERLAND,
            &[("open", false), ("read", true)],
        );
        fuzz(&mut combined);
    }

    // ---- Fat-slice selection (the shared `thin_slice_range`, host-tested) ----
    //
    // These exercise the fat-slice selector the live Darwin engine
    // (`crate::module::load_thin_slice`) uses. Previously they lived in
    // `module_macho.rs` and ran only on a macOS host; folding the selector into this
    // (host-independent) parser makes them run on any host — the check that would
    // have caught the original thin-slice-relative offset bug on this Windows host.

    fn put_be32(buf: &mut [u8], at: usize, v: u32) {
        buf[at..at + 4].copy_from_slice(&v.to_be_bytes());
    }
    fn put_be64(buf: &mut [u8], at: usize, v: u64) {
        buf[at..at + 8].copy_from_slice(&v.to_be_bytes());
    }
    fn put_le32(buf: &mut [u8], at: usize, v: u32) {
        buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// Assemble a synthetic universal container over `[(cputype, cpusubtype)]`. Each
    /// slice is laid out on its own page, begins with a thin `MH_MAGIC_64`, and is
    /// `0x100` bytes. `wide` selects the 64-bit `fat_arch_64` table. Returns the bytes
    /// and the byte range of each slice, in input order.
    fn build_fat(wide: bool, arches: &[(u32, u32)]) -> (Vec<u8>, Vec<core::ops::Range<usize>>) {
        let n = arches.len();
        let slice_size = 0x100usize;
        let mut buf = vec![0u8; 0x1000 + n * 0x1000];
        put_be32(&mut buf, 0, if wide { FAT_MAGIC_64 } else { FAT_MAGIC });
        put_be32(&mut buf, 4, n as u32);
        let entry_size = if wide { 32usize } else { 20usize };
        let mut ranges = Vec::new();
        for (i, &(ct, cs)) in arches.iter().enumerate() {
            let slice_off = 0x1000 + i * 0x1000;
            let entry = 8 + i * entry_size;
            put_be32(&mut buf, entry, ct);
            put_be32(&mut buf, entry + 4, cs);
            if wide {
                put_be64(&mut buf, entry + 8, slice_off as u64);
                put_be64(&mut buf, entry + 16, slice_size as u64);
                put_be32(&mut buf, entry + 24, 12); // align
            } else {
                put_be32(&mut buf, entry + 8, slice_off as u32);
                put_be32(&mut buf, entry + 12, slice_size as u32);
                put_be32(&mut buf, entry + 16, 12); // align
            }
            put_le32(&mut buf, slice_off, MH_MAGIC_64);
            ranges.push(slice_off..slice_off + slice_size);
        }
        (buf, ranges)
    }

    #[test]
    fn thin_file_is_the_whole_slice() {
        let mut buf = vec![0u8; 64];
        put_le32(&mut buf, 0, MH_MAGIC_64);
        assert_eq!(
            thin_slice_range(&buf, arch::CPU_TYPE_ARM64, 0).unwrap(),
            0..64
        );
    }

    #[test]
    fn fat32_selects_the_loaded_arch_slice() {
        let (buf, ranges) = build_fat(
            false,
            &[(arch::CPU_TYPE_X86_64, 0), (arch::CPU_TYPE_ARM64, 0)],
        );
        // Each arch resolves to ITS slice's offset — not file offset 0 (the old bug).
        assert_eq!(
            thin_slice_range(&buf, arch::CPU_TYPE_X86_64, 0).unwrap(),
            ranges[0]
        );
        let arm = thin_slice_range(&buf, arch::CPU_TYPE_ARM64, 0).unwrap();
        assert_eq!(arm, ranges[1]);
        assert_ne!(arm.start, 0, "the arm64 slice is at a nonzero fat offset");
        // The selected slice really begins with the thin Mach-O magic.
        assert_eq!(
            u32::from_le_bytes(buf[arm.start..arm.start + 4].try_into().unwrap()),
            MH_MAGIC_64,
        );
        // An architecture that is not present is a clean error, never a wrong slice.
        assert!(thin_slice_range(&buf, arch::CPU_TYPE_ARM64 + 1, 0).is_err());
    }

    #[test]
    fn fat64_selects_the_loaded_arch_slice() {
        let (buf, ranges) = build_fat(
            true,
            &[(arch::CPU_TYPE_ARM64, 0), (arch::CPU_TYPE_X86_64, 0)],
        );
        assert_eq!(
            thin_slice_range(&buf, arch::CPU_TYPE_ARM64, 0).unwrap(),
            ranges[0]
        );
        assert_eq!(
            thin_slice_range(&buf, arch::CPU_TYPE_X86_64, 0).unwrap(),
            ranges[1]
        );
    }

    #[test]
    fn arm64_and_arm64e_are_distinguished_by_subtype() {
        // Same cputype, different subtype: selecting by cputype alone would grab the
        // wrong slice and read garbage chain encodings.
        let (buf, ranges) = build_fat(
            false,
            &[
                (arch::CPU_TYPE_ARM64, 0),
                (arch::CPU_TYPE_ARM64, arch::CPU_SUBTYPE_ARM64E),
            ],
        );
        assert_eq!(
            thin_slice_range(&buf, arch::CPU_TYPE_ARM64, 0).unwrap(),
            ranges[0]
        );
        assert_eq!(
            thin_slice_range(&buf, arch::CPU_TYPE_ARM64, arch::CPU_SUBTYPE_ARM64E).unwrap(),
            ranges[1],
        );
    }

    #[test]
    fn subtype_capability_bits_are_masked_off() {
        // A `fat_arch.cpusubtype` may carry capability bits in its high byte; matching
        // masks them (like `is_arm64e`), so the loaded image's masked subtype matches.
        let cap = arch::CPU_SUBTYPE_ARM64E | arch::CPU_SUBTYPE_MASK;
        let (buf, ranges) = build_fat(false, &[(arch::CPU_TYPE_ARM64, cap)]);
        assert_eq!(
            thin_slice_range(&buf, arch::CPU_TYPE_ARM64, arch::CPU_SUBTYPE_ARM64E).unwrap(),
            ranges[0],
        );
    }

    #[test]
    fn out_of_bounds_or_non_macho_is_rejected_not_read() {
        // Neither a thin Mach-O nor a universal container.
        assert!(thin_slice_range(&[0u8; 64], arch::CPU_TYPE_ARM64, 0).is_err());
        // A fat slice whose offset runs past EOF is rejected, never indexed.
        let (mut buf, _) = build_fat(false, &[(arch::CPU_TYPE_ARM64, 0)]);
        put_be32(&mut buf, 16, 0xffff_fff0); // arch[0].offset well past the buffer
        assert!(thin_slice_range(&buf, arch::CPU_TYPE_ARM64, 0).is_err());
    }
}
