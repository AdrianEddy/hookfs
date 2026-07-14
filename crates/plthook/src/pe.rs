//!
//! This module parses the import tables of a **loaded** PE image — an image the
//! Windows loader has already mapped, so a relative virtual address (RVA) is
//! simply an offset from the module base and the on-disk section table is never
//! structure is decoded here and **every** offset, count, string, and slot
//! address is validated against the module's mapped range *before* the byte is
//! read. A malformed image therefore yields a typed [`Error`] and never a read
//! out of bounds.
//!
//! The parser is deliberately free of Windows syscalls so its logic can be unit-
//! and fuzz-tested on any host (see the tests at the bottom of this file); the
//! OS integration lives in [`crate::sys`] and [`crate::module`].

use crate::error::{Error, Result};
use crate::import::RawImport;
use crate::{ImportKind, Symbol};
use std::sync::Arc;

/// PE32+ ordinal flag: when set in an `IMAGE_THUNK_DATA`, the low 16 bits are an
/// ordinal rather than an RVA to an `IMAGE_IMPORT_BY_NAME`.
const IMAGE_ORDINAL_FLAG64: u64 = 0x8000_0000_0000_0000;

/// Optional-header magic for a 64-bit (PE32+) image.
const MAGIC_PE32_PLUS: u16 = 0x020B;
/// Optional-header magic for a 32-bit (PE32) image.
const MAGIC_PE32: u16 = 0x010B;

/// Data-directory index of the standard import table.
const DIRECTORY_ENTRY_IMPORT: usize = 1;
/// Data-directory index of the delay-load import table.
const DIRECTORY_ENTRY_DELAY_IMPORT: usize = 13;

/// Size of one `IMAGE_IMPORT_DESCRIPTOR`.
const IMPORT_DESCRIPTOR_SIZE: usize = 20;
/// Size of one `IMAGE_DELAYLOAD_DESCRIPTOR`.
const DELAY_DESCRIPTOR_SIZE: usize = 32;
/// Size of one PE32+ `IMAGE_THUNK_DATA` / IAT slot.
const THUNK_SIZE: usize = 8;

/// Defensive upper bounds. A conformant image stays far below these; exceeding
/// one means the tables are malformed (or maliciously unterminated), so the
/// parser stops with [`Error::Malformed`] instead of looping.
const MAX_DESCRIPTORS: usize = 8192;
const MAX_THUNKS: usize = 1 << 20;
const MAX_NAME_LEN: usize = 8192;

/// A contiguous, readable range of a mapped module: the "validated mapped range"
///
/// For a live module, `base` is the `HMODULE` and `size` is the loader-reported
/// `SizeOfImage` ([`crate::sys::image_size`]); the Windows loader commits the
/// whole `[base, base+size)` range, so reads within it are sound. For tests the
/// range is backed by an owned byte buffer.
pub(crate) struct MappedImage {
    base: usize,
    size: usize,
}

impl MappedImage {
    /// Wrap a loaded module's mapped range.
    ///
    /// # Safety
    /// `base` must be the base address of a module currently mapped into this
    /// process, and `size` must not exceed the number of readable bytes the
    /// loader committed there (its `SizeOfImage`). The range must remain mapped
    /// for the lifetime of this `MappedImage`.
    pub(crate) unsafe fn from_loaded(base: usize, size: usize) -> Self {
        Self { base, size }
    }

    /// Read `N` bytes at `offset`, bounds-checked against the mapped range.
    fn read<const N: usize>(&self, offset: usize) -> Result<[u8; N]> {
        let end = offset
            .checked_add(N)
            .ok_or(Error::Malformed("offset arithmetic overflow"))?;
        if end > self.size {
            return Err(Error::Malformed("read past end of mapped image"));
        }
        let addr = self
            .base
            .checked_add(offset)
            .ok_or(Error::Malformed("address arithmetic overflow"))?;
        let mut buf = [0u8; N];
        // SAFETY: the bounds check above proves `[addr, addr+N)` lies inside the
        // validated readable mapped range, and `buf` cannot overlap it.
        unsafe {
            core::ptr::copy_nonoverlapping(addr as *const u8, buf.as_mut_ptr(), N);
        }
        Ok(buf)
    }

    fn u16(&self, offset: usize) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read::<2>(offset)?))
    }

    fn u32_as_usize(&self, offset: usize) -> Result<usize> {
        // Widening on every supported (64-bit) host: no truncation.
        Ok(u32::from_le_bytes(self.read::<4>(offset)?) as usize)
    }

    fn thunk(&self, offset: usize) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read::<8>(offset)?))
    }

    /// Read a NUL-terminated ASCII string at `offset`, bounded by the mapped
    /// range and by [`MAX_NAME_LEN`]. Import names are ASCII; any stray non-UTF8
    /// byte is replaced rather than trusted.
    fn cstr(&self, offset: usize) -> Result<String> {
        let mut bytes = Vec::new();
        let mut cursor = offset;
        loop {
            let [byte] = self.read::<1>(cursor)?;
            if byte == 0 {
                break;
            }
            bytes.push(byte);
            if bytes.len() > MAX_NAME_LEN {
                return Err(Error::Malformed("unterminated import name string"));
            }
            cursor += 1;
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Absolute address (for patching) of an IAT thunk at `rva`.
    fn slot_address(&self, rva: usize) -> Result<usize> {
        // Validate the whole 8-byte slot lies in range before handing out its
        // address, so a caller can dereference/patch it safely.
        let end = rva
            .checked_add(THUNK_SIZE)
            .ok_or(Error::Malformed("slot arithmetic overflow"))?;
        if end > self.size {
            return Err(Error::Malformed("import slot past end of mapped image"));
        }
        // Alignment: on the install/restore path this address is reinterpreted as
        // an `&AtomicUsize` (a pointer-width atomic) and swapped/loaded, so a
        // misaligned slot would be instant undefined behavior — an alignment
        // fault on aarch64 (`LDXR`) and a possible split-lock `#AC` on x86-64.
        // The Windows loader tolerates a misaligned `FirstThunk` /
        // `ImportAddressTableRVA` in a crafted-but-loadable image, so this choke
        // point — the sole producer of every patchable slot address, shared by
        // `install` and `restore` — must reject it with a typed error. The module
        // base is page-aligned, so a pointer-aligned RVA yields a pointer-aligned
        // absolute address.
        if !rva.is_multiple_of(core::mem::align_of::<usize>()) {
            return Err(Error::Malformed("import slot is not pointer-aligned"));
        }
        self.base
            .checked_add(rva)
            .ok_or(Error::Malformed("slot address overflow"))
    }

    /// Resolve a delay-import descriptor address field to an RVA (offset from the
    /// module base), honoring the `Attributes.RvaBased` interpretation
    ///
    /// When `uses_rva` the field is already an RVA and is returned unchanged (the
    /// modern form MSVC has emitted since VS2015). Otherwise it is an absolute VA —
    /// a legacy pre-VS2015 delay descriptor whose fields the loader base-relocated
    /// in place — so it is converted back to an RVA and validated to land inside the
    /// mapped image. `IMAGE_DELAYLOAD_DESCRIPTOR` fields are 32-bit, so an absolute
    /// VA cannot address a module mapped above 4 GiB: in a 64-bit image (the only
    /// kind this engine loads) the absolute-VA form is never legitimate and is
    /// rejected here with a typed error rather than dereferenced as a truncated
    /// pointer — but it is *interpreted*, not blanket-refused, so a valid image is
    /// never denied over a form it does not actually use.
    fn field_rva(&self, field: usize, uses_rva: bool) -> Result<usize> {
        if uses_rva {
            return Ok(field);
        }
        let rva = field.checked_sub(self.base).ok_or(Error::Malformed(
            "delay-import absolute VA below module base",
        ))?;
        if rva >= self.size {
            return Err(Error::Malformed(
                "delay-import absolute VA outside mapped image",
            ));
        }
        Ok(rva)
    }
}

/// RVAs of the standard/delay import data directories within the mapped image,
/// resolved from the validated optional header. Only the directory *address* is
/// retained: the directory `Size` is intentionally not carried, because it is not
/// a sound bound for the descriptor walk (see [`parse_standard`]).
struct Directories {
    import: Option<usize>,
    delay: Option<usize>,
}

/// Validate the DOS/NT/optional headers and return the import data-directory
/// locations. Rejects PE32 (32-bit) images inside a 64-bit process and any
/// header field that falls outside the mapped range.
fn parse_headers(image: &MappedImage) -> Result<Directories> {
    // IMAGE_DOS_HEADER: "MZ" magic, then e_lfanew at offset 0x3C.
    if image.u16(0)? != 0x5A4D {
        return Err(Error::Malformed("missing 'MZ' DOS signature"));
    }
    let e_lfanew = image.u32_as_usize(0x3C)?;

    // IMAGE_NT_HEADERS: "PE\0\0" signature.
    if image.read::<4>(e_lfanew)? != *b"PE\0\0" {
        return Err(Error::Malformed("missing 'PE\\0\\0' NT signature"));
    }

    // IMAGE_FILE_HEADER (immediately after the 4-byte signature).
    let file_header = e_lfanew
        .checked_add(4)
        .ok_or(Error::Malformed("NT header offset overflow"))?;
    let machine = image.u16(file_header)?;
    // `IMAGE_FILE_HEADER.SizeOfOptionalHeader` is at offset +16 (winnt.h): after
    // Machine(+0), NumberOfSections(+2), TimeDateStamp(+4), PointerToSymbolTable
    // (+8), NumberOfSymbols(+12). (Offset +20 is the optional header's `Magic`,
    // always 0x20B/0x10B, which would defeat the `< 112` guard below.)
    let size_of_optional = image.u16(file_header + 16)? as usize;

    // IMAGE_OPTIONAL_HEADER begins immediately after the 20-byte file header.
    let optional = file_header
        .checked_add(20)
        .ok_or(Error::Malformed("optional header offset overflow"))?;
    if size_of_optional < 112 {
        return Err(Error::Malformed("optional header too small"));
    }
    let magic = image.u16(optional)?;
    match magic {
        MAGIC_PE32_PLUS => {}
        MAGIC_PE32 => {
            return Err(Error::Unsupported(
                "PE32 (32-bit) image; only PE32+ is supported at runtime",
            ));
        }
        _ => return Err(Error::Malformed("unrecognized optional-header magic")),
    }
    // A live module always matches the host machine; reject a mismatch as a sign
    // the base/size are wrong rather than parsing on.
    if machine != crate::arch::HOST_MACHINE {
        return Err(Error::Unsupported(
            "image machine does not match host process",
        ));
    }

    // Data directories begin at optional + 112 in PE32+; each entry is
    // { VirtualAddress: u32, Size: u32 }.
    let number_of_rva_and_sizes = image.u32_as_usize(optional + 108)?;
    let data_dir_base = optional + 112;

    let directory = |index: usize| -> Result<Option<usize>> {
        if index >= number_of_rva_and_sizes {
            return Ok(None);
        }
        let entry = data_dir_base + index * 8;
        let rva = image.u32_as_usize(entry)?;
        // The directory `Size` (at `entry + 4`) is deliberately not read: the
        // descriptor walk is bounded by the null sentinel, `SizeOfImage`, and
        // `MAX_DESCRIPTORS`, never by `Size` (see `parse_standard`).
        Ok((rva != 0).then_some(rva))
    };

    Ok(Directories {
        import: directory(DIRECTORY_ENTRY_IMPORT)?,
        delay: directory(DIRECTORY_ENTRY_DELAY_IMPORT)?,
    })
}

/// Validate the DOS/NT/optional headers of a mapped image without enumerating
/// imports — used at module acquisition to reject an invalid handle/address
/// early.
pub(crate) fn validate(image: &MappedImage) -> Result<()> {
    parse_headers(image).map(|_| ())
}

/// Enumerate every import slot of a loaded PE image (standard + delay-load).
pub(crate) fn parse_imports(image: &MappedImage) -> Result<Vec<RawImport>> {
    let dirs = parse_headers(image)?;
    let mut out = Vec::new();
    if let Some(rva) = dirs.import {
        // Standard imports are the primary hook target (kernel32 file I/O, the
        // decoder DLLs' CRT imports): a malformed *standard* table is a hard error.
        parse_standard(image, rva, &mut out)?;
    }
    if let Some(rva) = dirs.delay {
        // Delay imports are a secondary, optional directory — many modules have
        // none. A defect there (an unsupported/legacy descriptor, a corrupt table)
        // must NOT discard the standard imports already collected, which are what
        // callers actually hook. So the delay walk fills a scratch buffer and is
        // merged only on success; on failure the standard imports are preserved and
        // the delay contribution is dropped, rather than failing the whole module.
        // (Before this, a single legacy/corrupt delay descriptor made the entire
        // module — including every valid standard import — unhookable.)
        let mut delay = Vec::new();
        if parse_delay(image, rva, &mut delay).is_ok() {
            out.append(&mut delay);
        }
    }
    Ok(out)
}

/// Walk `IMAGE_DIRECTORY_ENTRY_IMPORT`.
///
/// The `IMAGE_IMPORT_DESCRIPTOR` array is walked to its authoritative terminator —
/// the null (all-zero) descriptor, canonically detected by `Name == 0` — exactly
/// as the Windows loader does. The data-directory `Size` is deliberately **not**
/// used to bound the walk: a linker that under-sizes it would otherwise truncate
/// enumeration before the null sentinel. Safety instead rests on two
/// `Size`-independent guards — every field is bounds-checked against `SizeOfImage`
/// before it is read (so the walk can never leave the mapped image), and
/// [`MAX_DESCRIPTORS`] caps a runaway walk of an image whose terminator is missing
/// or corrupt (which is then reported as [`Error::Malformed`]).
fn parse_standard(image: &MappedImage, dir_rva: usize, out: &mut Vec<RawImport>) -> Result<()> {
    for i in 0..MAX_DESCRIPTORS {
        let desc = dir_rva + i * IMPORT_DESCRIPTOR_SIZE;
        let original_first_thunk = image.u32_as_usize(desc)?;
        let name_rva = image.u32_as_usize(desc + 12)?;
        let first_thunk = image.u32_as_usize(desc + 16)?;
        // The array is terminated by an all-zero descriptor; `Name == 0` is the
        // canonical sentinel.
        if name_rva == 0 {
            return Ok(());
        }
        if first_thunk == 0 {
            // No IAT to patch — nothing enumerable; skip defensively.
            continue;
        }
        let library: Arc<str> = image.cstr(name_rva)?.into();
        collect_thunks(
            image,
            original_first_thunk,
            first_thunk,
            &library,
            ImportKind::Standard,
            out,
        )?;
    }
    Err(Error::Malformed(
        "import descriptor table is not NUL-terminated",
    ))
}

/// Walk `IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT`.
///
/// Walked to the null (all-zero) `IMAGE_DELAYLOAD_DESCRIPTOR` sentinel (canonically
/// `DllNameRVA == 0`), on the same `Size`-independent basis as [`parse_standard`]:
/// the directory `Size` never truncates the walk; `SizeOfImage` bounds every read
/// and [`MAX_DESCRIPTORS`] guards runaway iteration.
fn parse_delay(image: &MappedImage, dir_rva: usize, out: &mut Vec<RawImport>) -> Result<()> {
    for i in 0..MAX_DESCRIPTORS {
        let desc = dir_rva + i * DELAY_DESCRIPTOR_SIZE;
        let attributes = image.u32_as_usize(desc)?;
        let dll_name_rva = image.u32_as_usize(desc + 4)?;
        let slots_field = image.u32_as_usize(desc + 12)?;
        let names_field = image.u32_as_usize(desc + 16)?;
        if dll_name_rva == 0 {
            return Ok(());
        }
        // Bit 0 (`DLATTR_RVA`) selects how the descriptor's address fields are
        // interpreted: set → RVAs (the only form MSVC has emitted since VS2015);
        // for why the absolute-VA form is inherently 32-bit-only and thus rejected
        // (not dereferenced) in a 64-bit image. The DLL-name field is treated as an
        // RVA in both forms, matching the reference plthook.
        let uses_rva = attributes & 0x1 != 0;
        let library: Arc<str> = image.cstr(dll_name_rva)?.into();
        if library.is_empty() {
            continue;
        }
        if slots_field == 0 {
            continue;
        }
        // A zero INT field means "no import name table" (names unrecoverable from a
        // loaded image); pass it through as `0` so `collect_thunks` takes its
        // address-only path rather than resolving a bogus VA.
        let names_rva = if names_field == 0 {
            0
        } else {
            image.field_rva(names_field, uses_rva)?
        };
        let slots_rva = image.field_rva(slots_field, uses_rva)?;
        collect_thunks(
            image,
            names_rva,
            slots_rva,
            &library,
            ImportKind::DelayLoad,
            out,
        )?;
    }
    Err(Error::Malformed(
        "delay-import descriptor table is not NUL-terminated",
    ))
}

/// Walk a parallel INT (names) / IAT (slots) pair for one imported library.
///
/// When `int_rva == 0` the names are unrecoverable from a loaded image (the
/// loader overwrote the IAT with resolved addresses), so the IAT is enumerated
/// by its NUL terminator and each slot is exposed as address-only
/// (`symbol == None`).
fn collect_thunks(
    image: &MappedImage,
    names_rva: usize,
    slots_rva: usize,
    library: &Arc<str>,
    kind: ImportKind,
    out: &mut Vec<RawImport>,
) -> Result<()> {
    if names_rva == 0 {
        for j in 0..MAX_THUNKS {
            let entry_rva = slots_rva + j * THUNK_SIZE;
            if image.thunk(entry_rva)? == 0 {
                return Ok(());
            }
            out.push(RawImport {
                library: Arc::clone(library),
                symbol: None,
                version: None,
                slot: image.slot_address(entry_rva)?,
                kind,
                authenticated: false,
            });
        }
        return Err(Error::Malformed(
            "import address table is not NUL-terminated",
        ));
    }

    let is_winsock2 = library.eq_ignore_ascii_case("ws2_32.dll");
    for j in 0..MAX_THUNKS {
        let name_entry = image.thunk(names_rva + j * THUNK_SIZE)?;
        if name_entry == 0 {
            return Ok(());
        }
        let symbol = if name_entry & IMAGE_ORDINAL_FLAG64 != 0 {
            // Import by ordinal: the low 16 bits.
            let ordinal = (name_entry & 0xFFFF) as u16;
            // `WS2_32.DLL` exports many entry points by ordinal only (a stable,
            // documented ABI). The reference plthook maps the well-known ordinals
            // back to their canonical names so an ordinal-bound import such as
            // `connect` (ordinal 4) is enumerated — and therefore hookable — by
            match winsock2_ordinal2name(ordinal).filter(|_| is_winsock2) {
                Some(name) => Symbol::Name(name.to_owned()),
                None => Symbol::Ordinal(ordinal),
            }
        } else {
            // Import by name: RVA to IMAGE_IMPORT_BY_NAME { hint: u16, name }.
            let by_name_rva = (name_entry & !IMAGE_ORDINAL_FLAG64) as usize;
            Symbol::Name(image.cstr(by_name_rva + 2)?)
        };
        out.push(RawImport {
            library: Arc::clone(library),
            symbol: Some(symbol),
            version: None,
            slot: image.slot_address(slots_rva + j * THUNK_SIZE)?,
            kind,
            authenticated: false,
        });
    }
    Err(Error::Malformed("import name table is not NUL-terminated"))
}

/// Map a well-known `WS2_32.DLL` export ordinal to its canonical function name.
///
/// Winsock2 exports a large set of functions by ordinal only; those ordinals are a
/// fixed, documented ABI (unchanged since the Winsock 1.1/2.0 era). Ported verbatim
/// from the reference `plthook_win32.c` so that a program importing e.g. `connect`
/// from `ws2_32.dll` *by ordinal* still enumerates by name and can be hooked with a
/// name-based replacement. Returns `None` for any ordinal outside the known set (the
/// caller then falls back to [`Symbol::Ordinal`]).
fn winsock2_ordinal2name(ordinal: u16) -> Option<&'static str> {
    Some(match ordinal {
        1 => "accept",
        2 => "bind",
        3 => "closesocket",
        4 => "connect",
        5 => "getpeername",
        6 => "getsockname",
        7 => "getsockopt",
        8 => "htonl",
        9 => "htons",
        10 => "inet_addr",
        11 => "inet_ntoa",
        12 => "ioctlsocket",
        13 => "listen",
        14 => "ntohl",
        15 => "ntohs",
        16 => "recv",
        17 => "recvfrom",
        18 => "select",
        19 => "send",
        20 => "sendto",
        21 => "setsockopt",
        22 => "shutdown",
        23 => "socket",
        24 => "MigrateWinsockConfiguration",
        51 => "gethostbyaddr",
        52 => "gethostbyname",
        53 => "getprotobyname",
        54 => "getprotobynumber",
        55 => "getservbyname",
        56 => "getservbyport",
        57 => "gethostname",
        101 => "WSAAsyncSelect",
        102 => "WSAAsyncGetHostByAddr",
        103 => "WSAAsyncGetHostByName",
        104 => "WSAAsyncGetProtoByNumber",
        105 => "WSAAsyncGetProtoByName",
        106 => "WSAAsyncGetServByPort",
        107 => "WSAAsyncGetServByName",
        108 => "WSACancelAsyncRequest",
        109 => "WSASetBlockingHook",
        110 => "WSAUnhookBlockingHook",
        111 => "WSAGetLastError",
        112 => "WSASetLastError",
        113 => "WSACancelBlockingCall",
        114 => "WSAIsBlocking",
        115 => "WSAStartup",
        116 => "WSACleanup",
        151 => "__WSAFDIsSet",
        500 => "WEP",
        1000 => "WSApSetPostRoutine",
        1001 => "WsControl",
        1002 => "closesockinfo",
        1003 => "Arecv",
        1004 => "Asend",
        1005 => "WSHEnumProtocols",
        1100 => "inet_network",
        1101 => "getnetbyname",
        1102 => "rcmd",
        1103 => "rexec",
        1104 => "rresvport",
        1105 => "sethostname",
        1106 => "dn_expand",
        1107 => "WSARecvEx",
        1108 => "s_perror",
        1109 => "GetAddressByNameA",
        1110 => "GetAddressByNameW",
        1111 => "EnumProtocolsA",
        1112 => "EnumProtocolsW",
        1113 => "GetTypeByNameA",
        1114 => "GetTypeByNameW",
        1115 => "GetNameByTypeA",
        1116 => "GetNameByTypeW",
        1117 => "SetServiceA",
        1118 => "SetServiceW",
        1119 => "GetServiceA",
        1120 => "GetServiceW",
        1130 => "NPLoadNameSpaces",
        1131 => "NSPStartup",
        1140 => "TransmitFile",
        1141 => "AcceptEx",
        1142 => "GetAcceptExSockaddrs",
        _ => return None,
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::cast_possible_truncation,
    clippy::panic,
    // INT/IAT are the canonical PE abbreviations; their similarity is not a smell.
    clippy::similar_names
)]
mod tests {
    use super::*;

    /// Builds a **loaded-image-layout** PE32+ buffer: byte offset == RVA, so the
    /// parser (which treats the buffer start as the module base) reads every RVA
    /// directly, exactly as it would from a mapped module. No section table is
    /// needed because a loaded image never uses one.
    struct Builder {
        buf: Vec<u8>,
    }

    /// Bytes reserved at the front for DOS + NT + optional headers.
    const HEADER_AREA: usize = 0x200;
    const E_LFANEW: usize = 0x80;
    const OPTIONAL: usize = E_LFANEW + 4 + 20;
    const DATA_DIR: usize = OPTIONAL + 112;

    impl Builder {
        fn new() -> Self {
            Self {
                buf: vec![0u8; HEADER_AREA],
            }
        }
        fn rva(&self) -> u32 {
            self.buf.len() as u32
        }
        fn align(&mut self, n: usize) {
            while !self.buf.len().is_multiple_of(n) {
                self.buf.push(0);
            }
        }
        fn u16(&mut self, v: u16) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        fn u32(&mut self, v: u32) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        fn u64(&mut self, v: u64) {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
        fn cstr(&mut self, s: &str) -> u32 {
            let rva = self.rva();
            self.buf.extend_from_slice(s.as_bytes());
            self.buf.push(0);
            rva
        }
        fn hint_name(&mut self, hint: u16, name: &str) -> u32 {
            self.align(2);
            let rva = self.rva();
            self.u16(hint);
            self.buf.extend_from_slice(name.as_bytes());
            self.buf.push(0);
            rva
        }
        fn poke(&mut self, off: usize, bytes: &[u8]) {
            self.buf[off..off + bytes.len()].copy_from_slice(bytes);
        }
        /// Write the fixed headers and the two import data-directory entries,
        /// then hand back the finished image bytes.
        fn finish(mut self, import_dir: (u32, u32), delay_dir: (u32, u32)) -> Vec<u8> {
            let size_of_image = self.buf.len() as u32;
            self.poke(0, b"MZ");
            self.poke(0x3C, &(E_LFANEW as u32).to_le_bytes());
            self.poke(E_LFANEW, b"PE\0\0");
            // Match the host machine so the header validation accepts the fixture
            // on both x86_64 and aarch64 hosts.
            self.poke(E_LFANEW + 4, &crate::arch::HOST_MACHINE.to_le_bytes());
            self.poke(E_LFANEW + 4 + 16, &240u16.to_le_bytes()); // SizeOfOptionalHeader
            self.poke(OPTIONAL, &MAGIC_PE32_PLUS.to_le_bytes());
            self.poke(OPTIONAL + 56, &size_of_image.to_le_bytes()); // SizeOfImage
            self.poke(OPTIONAL + 108, &16u32.to_le_bytes()); // NumberOfRvaAndSizes
            self.poke(
                DATA_DIR + DIRECTORY_ENTRY_IMPORT * 8,
                &import_dir.0.to_le_bytes(),
            );
            self.poke(
                DATA_DIR + DIRECTORY_ENTRY_IMPORT * 8 + 4,
                &import_dir.1.to_le_bytes(),
            );
            self.poke(
                DATA_DIR + DIRECTORY_ENTRY_DELAY_IMPORT * 8,
                &delay_dir.0.to_le_bytes(),
            );
            self.poke(
                DATA_DIR + DIRECTORY_ENTRY_DELAY_IMPORT * 8 + 4,
                &delay_dir.1.to_le_bytes(),
            );
            self.buf
        }
    }

    // On a non-Windows dev host `crate::arch::HOST_MACHINE` is still AMD64 for
    // x86_64, matching the fixtures. The parser tests run wherever the crate
    // compiles for x86_64/aarch64.
    fn parse(bytes: &[u8]) -> Result<Vec<RawImport>> {
        // SAFETY: the byte buffer backs the whole mapped range for its lifetime.
        let image = unsafe { MappedImage::from_loaded(bytes.as_ptr() as usize, bytes.len()) };
        parse_imports(&image)
    }

    #[test]
    fn standard_named_and_ordinal_imports() {
        let mut b = Builder::new();
        // Import descriptor with placeholders, back-patched once RVAs are known.
        let desc_rva = b.rva();
        for _ in 0..5 {
            b.u32(0);
        }
        b.u32(0); // fields spill into second (terminator) descriptor
        for _ in 0..4 {
            b.u32(0);
        }
        let name_rva = b.cstr("KERNEL32.dll");
        let create = b.hint_name(0x0100, "CreateFileW");
        b.align(8);
        let int_rva = b.rva();
        b.u64(u64::from(create)); // INT[0]: name
        b.u64(IMAGE_ORDINAL_FLAG64 | 7); // INT[1]: ordinal #7
        b.u64(0); // terminator
        let iat_rva = b.rva();
        b.u64(0x1111); // IAT[0]: resolved address (loaded image)
        b.u64(0x2222); // IAT[1]
        b.u64(0); // terminator
        b.poke(desc_rva as usize, &int_rva.to_le_bytes()); // OriginalFirstThunk
        b.poke(desc_rva as usize + 12, &name_rva.to_le_bytes()); // Name
        b.poke(desc_rva as usize + 16, &iat_rva.to_le_bytes()); // FirstThunk

        let bytes = b.finish((desc_rva, 40), (0, 0));
        let base = bytes.as_ptr() as usize;
        let imports = parse(&bytes).expect("well-formed image parses");
        assert_eq!(imports.len(), 2);

        assert_eq!(&*imports[0].library, "KERNEL32.dll");
        assert_eq!(imports[0].symbol, Some(Symbol::Name("CreateFileW".into())));
        assert_eq!(imports[0].kind, ImportKind::Standard);
        assert_eq!(imports[0].slot, base + iat_rva as usize); // &IAT[0]

        assert_eq!(imports[1].symbol, Some(Symbol::Ordinal(7)));
        assert_eq!(imports[1].slot, base + iat_rva as usize + 8); // &IAT[1]
    }

    #[test]
    fn undersized_import_directory_size_enumerates_to_null_terminator() {
        // Two import descriptors followed by a null terminator, but the data-
        // directory `Size` is set to a single descriptor (20 bytes) — smaller than
        // the real array. The Windows loader ignores `Size` and walks to the null
        // sentinel; the engine must do the same and enumerate BOTH libraries. A
        // `Size`-capped walk (`descriptor_cap(20, 20) == 1`, the prior behavior)
        // would stop after the first descriptor and miss the second.
        let mut b = Builder::new();
        let desc_rva = b.rva();
        // Reserve three descriptors (2 real + terminator); back-patched below.
        for _ in 0..(3 * 5) {
            b.u32(0);
        }
        // Library A: KERNEL32.dll importing CreateFileW.
        let name_a = b.cstr("KERNEL32.dll");
        let func_a = b.hint_name(0, "CreateFileW");
        b.align(8);
        let int_a = b.rva();
        b.u64(u64::from(func_a));
        b.u64(0);
        let iat_a = b.rva();
        b.u64(0x1111);
        b.u64(0);
        // Library B: USER32.dll importing MessageBoxW.
        let name_b = b.cstr("USER32.dll");
        let func_b = b.hint_name(0, "MessageBoxW");
        b.align(8);
        let int_b = b.rva();
        b.u64(u64::from(func_b));
        b.u64(0);
        let iat_b = b.rva();
        b.u64(0x2222);
        b.u64(0);
        // Descriptor 0 → library A.
        b.poke(desc_rva as usize, &int_a.to_le_bytes());
        b.poke(desc_rva as usize + 12, &name_a.to_le_bytes());
        b.poke(desc_rva as usize + 16, &iat_a.to_le_bytes());
        // Descriptor 1 → library B.
        b.poke(desc_rva as usize + 20, &int_b.to_le_bytes());
        b.poke(desc_rva as usize + 20 + 12, &name_b.to_le_bytes());
        b.poke(desc_rva as usize + 20 + 16, &iat_b.to_le_bytes());
        // Descriptor 2 stays all-zero → the null terminator.

        // Directory `Size` = one descriptor: deliberately under-sized.
        let bytes = b.finish((desc_rva, IMPORT_DESCRIPTOR_SIZE as u32), (0, 0));
        let imports = parse(&bytes).expect("under-sized directory still parses");
        assert_eq!(
            imports.len(),
            2,
            "both descriptors enumerated past the under-sized Size"
        );
        assert_eq!(&*imports[0].library, "KERNEL32.dll");
        assert_eq!(imports[0].symbol, Some(Symbol::Name("CreateFileW".into())));
        assert_eq!(&*imports[1].library, "USER32.dll");
        assert_eq!(imports[1].symbol, Some(Symbol::Name("MessageBoxW".into())));
    }

    #[test]
    fn original_first_thunk_zero_is_address_only() {
        let mut b = Builder::new();
        let desc_rva = b.rva();
        for _ in 0..10 {
            b.u32(0);
        }
        let name_rva = b.cstr("LEGACY.dll");
        b.align(8);
        let iat_rva = b.rva();
        b.u64(0xAAAA); // resolved address (name RVAs already overwritten)
        b.u64(0xBBBB);
        b.u64(0); // terminator
        // OriginalFirstThunk stays 0; only Name and FirstThunk are set.
        b.poke(desc_rva as usize + 12, &name_rva.to_le_bytes());
        b.poke(desc_rva as usize + 16, &iat_rva.to_le_bytes());

        let bytes = b.finish((desc_rva, 40), (0, 0));
        let imports = parse(&bytes).expect("OFT==0 image parses");
        assert_eq!(imports.len(), 2);
        // Names are unrecoverable from a loaded image; slots are still exposed.
        assert!(imports.iter().all(|i| i.symbol.is_none()));
        assert!(imports.iter().all(|i| &*i.library == "LEGACY.dll"));
    }

    #[test]
    fn delay_import_rva_based() {
        let mut b = Builder::new();
        let desc_rva = b.rva();
        for _ in 0..8 {
            b.u32(0);
        }
        b.u32(0); // extra zeros → acts as terminator descriptor (DllNameRVA==0)
        for _ in 0..7 {
            b.u32(0);
        }
        let name_rva = b.cstr("DELAYED.dll");
        let func = b.hint_name(0, "DelayedFunc");
        b.align(8);
        let int_rva = b.rva();
        b.u64(u64::from(func));
        b.u64(0);
        let iat_rva = b.rva();
        b.u64(0xDEAD); // delay stub thunk (loaded, pre-resolution)
        b.u64(0);
        b.poke(desc_rva as usize, &1u32.to_le_bytes()); // Attributes: RvaBased
        b.poke(desc_rva as usize + 4, &name_rva.to_le_bytes()); // DllNameRVA
        b.poke(desc_rva as usize + 12, &iat_rva.to_le_bytes()); // ImportAddressTableRVA
        b.poke(desc_rva as usize + 16, &int_rva.to_le_bytes()); // ImportNameTableRVA

        let bytes = b.finish((0, 0), (desc_rva, 64));
        let imports = parse(&bytes).expect("delay-import image parses");
        assert_eq!(imports.len(), 1);
        assert_eq!(&*imports[0].library, "DELAYED.dll");
        assert_eq!(imports[0].symbol, Some(Symbol::Name("DelayedFunc".into())));
        assert_eq!(imports[0].kind, ImportKind::DelayLoad);
    }

    #[test]
    fn field_rva_interprets_rva_and_va() {
        // returned unchanged. VA form: converted to an offset and range-checked.
        let buf = vec![0u8; 0x1000];
        let base = buf.as_ptr() as usize;
        // SAFETY: `buf` backs the whole mapped range for its lifetime.
        let image = unsafe { MappedImage::from_loaded(base, buf.len()) };

        // RVA form: value is already an offset from base.
        assert_eq!(image.field_rva(0x500, true).unwrap(), 0x500);
        // VA form, in range: absolute VA → offset.
        assert_eq!(image.field_rva(base + 0x40, false).unwrap(), 0x40);
        assert_eq!(image.field_rva(base + 0xFFF, false).unwrap(), 0xFFF);
        // VA form, below base or at/after the end: rejected, never dereferenced.
        assert!(image.field_rva(base - 1, false).is_err());
        assert!(image.field_rva(base + 0x1000, false).is_err());
    }

    #[test]
    fn delay_failure_preserves_standard_imports() {
        // A valid standard import PLUS a broken (absolute-VA) delay directory. The
        // delay defect must NOT discard the standard import — which is what callers
        // actually hook. (Before the isolation fix, a single legacy/corrupt delay
        // descriptor made the whole module, standard imports included, unhookable.)
        let mut b = Builder::new();
        // Standard descriptor: KERNEL32.dll / CreateFileW.
        let std_desc = b.rva();
        for _ in 0..10 {
            b.u32(0); // one descriptor + terminator
        }
        let kname = b.cstr("KERNEL32.dll");
        let create = b.hint_name(0, "CreateFileW");
        b.align(8);
        let std_int = b.rva();
        b.u64(u64::from(create));
        b.u64(0);
        let std_iat = b.rva();
        b.u64(0x1111);
        b.u64(0);
        b.poke(std_desc as usize, &std_int.to_le_bytes());
        b.poke(std_desc as usize + 12, &kname.to_le_bytes());
        b.poke(std_desc as usize + 16, &std_iat.to_le_bytes());
        // Delay descriptor: absolute-VA form (Attributes==0) with a non-zero IAT
        // field. In this 64-bit fixture a 32-bit "VA" cannot address the image, so
        // `field_rva` rejects it and the delay directory is dropped whole.
        let delay_desc = b.rva();
        for _ in 0..8 {
            b.u32(0); // descriptor
        }
        for _ in 0..8 {
            b.u32(0); // terminator descriptor
        }
        let oldname = b.cstr("OLD.dll");
        b.poke(delay_desc as usize + 4, &oldname.to_le_bytes()); // DllNameRVA
        b.poke(delay_desc as usize + 12, &0x100u32.to_le_bytes()); // bogus absolute-VA IAT

        let bytes = b.finish((std_desc, 40), (delay_desc, 64));
        let imports = parse(&bytes).expect("standard imports survive a broken delay directory");
        assert_eq!(
            imports.len(),
            1,
            "delay contribution dropped, standard import kept"
        );
        assert_eq!(&*imports[0].library, "KERNEL32.dll");
        assert_eq!(imports[0].symbol, Some(Symbol::Name("CreateFileW".into())));
        assert_eq!(imports[0].kind, ImportKind::Standard);
    }

    #[test]
    fn winsock2_ordinal_resolves_to_name() {
        // WS2_32.dll imported by ordinal: the well-known ordinals resolve to their
        // canonical names (matching the reference plthook) so they enumerate — and
        // are hookable — by name; unknown ordinals fall back to `Symbol::Ordinal`.
        // The DLL-name match is case-insensitive ("WS2_32" vs "ws2_32").
        let mut b = Builder::new();
        let desc_rva = b.rva();
        for _ in 0..10 {
            b.u32(0);
        }
        let name_rva = b.cstr("WS2_32.dll");
        b.align(8);
        let int_rva = b.rva();
        let unknown_ordinal: u64 = 8888; // not in the winsock table
        b.u64(IMAGE_ORDINAL_FLAG64 | 4); // connect
        b.u64(IMAGE_ORDINAL_FLAG64 | unknown_ordinal);
        b.u64(0);
        let iat_rva = b.rva();
        b.u64(0x1111);
        b.u64(0x2222);
        b.u64(0);
        b.poke(desc_rva as usize, &int_rva.to_le_bytes());
        b.poke(desc_rva as usize + 12, &name_rva.to_le_bytes());
        b.poke(desc_rva as usize + 16, &iat_rva.to_le_bytes());
        let bytes = b.finish((desc_rva, 40), (0, 0));
        let imports = parse(&bytes).expect("winsock image parses");
        assert_eq!(imports.len(), 2);
        assert_eq!(imports[0].symbol, Some(Symbol::Name("connect".into())));
        assert_eq!(imports[1].symbol, Some(Symbol::Ordinal(8888)));
    }

    #[test]
    fn non_winsock_ordinal_is_not_named() {
        // The ordinal→name substitution is scoped to WS2_32.dll: the same ordinal
        // imported from another library stays an ordinal.
        let mut b = Builder::new();
        let desc_rva = b.rva();
        for _ in 0..10 {
            b.u32(0);
        }
        let name_rva = b.cstr("OLEAUT32.dll");
        b.align(8);
        let int_rva = b.rva();
        b.u64(IMAGE_ORDINAL_FLAG64 | 4);
        b.u64(0);
        let iat_rva = b.rva();
        b.u64(0x1111);
        b.u64(0);
        b.poke(desc_rva as usize, &int_rva.to_le_bytes());
        b.poke(desc_rva as usize + 12, &name_rva.to_le_bytes());
        b.poke(desc_rva as usize + 16, &iat_rva.to_le_bytes());
        let bytes = b.finish((desc_rva, 40), (0, 0));
        let imports = parse(&bytes).expect("image parses");
        assert_eq!(imports.len(), 1);
        assert_eq!(imports[0].symbol, Some(Symbol::Ordinal(4)));
    }

    #[test]
    fn bad_dos_signature_is_malformed() {
        // All-zero header area: no "MZ" DOS magic.
        let bytes = vec![0u8; HEADER_AREA];
        assert!(matches!(parse(&bytes), Err(Error::Malformed(_))));
    }

    #[test]
    fn validate_accepts_a_good_header_and_rejects_a_bad_one() {
        let good = Builder::new().finish((0, 0), (0, 0));
        // SAFETY: the byte buffer backs the mapped range for its lifetime.
        let image = unsafe { MappedImage::from_loaded(good.as_ptr() as usize, good.len()) };
        assert!(
            validate(&image).is_ok(),
            "a well-formed PE32+ header validates"
        );

        let bad = vec![0u8; HEADER_AREA];
        // SAFETY: as above.
        let image = unsafe { MappedImage::from_loaded(bad.as_ptr() as usize, bad.len()) };
        assert!(validate(&image).is_err(), "an all-zero header is rejected");
    }

    #[test]
    fn out_of_range_e_lfanew_is_malformed() {
        let mut bytes = Builder::new().finish((0, 0), (0, 0));
        // e_lfanew points far past the buffer.
        bytes[0x3C..0x40].copy_from_slice(&0x00FF_FFFFu32.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(Error::Malformed(_))));
    }

    #[test]
    fn pe32_is_unsupported() {
        let mut bytes = Builder::new().finish((0, 0), (0, 0));
        bytes[OPTIONAL..OPTIONAL + 2].copy_from_slice(&MAGIC_PE32.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(Error::Unsupported(_))));
    }

    #[test]
    fn import_directory_past_end_is_malformed() {
        // Directory RVA beyond SizeOfImage → first descriptor read is rejected.
        let bytes = Builder::new().finish((0x00FF_FFFF, 40), (0, 0));
        assert!(matches!(parse(&bytes), Err(Error::Malformed(_))));
    }

    #[test]
    fn truncated_thunk_table_is_malformed() {
        let mut b = Builder::new();
        let desc_rva = b.rva();
        for _ in 0..10 {
            b.u32(0);
        }
        let name_rva = b.cstr("KERNEL32.dll");
        b.align(8);
        let iat_rva = b.rva();
        // Non-terminated IAT: fill to end of buffer with non-zero, no NUL thunk.
        b.u64(0x1111);
        b.poke(desc_rva as usize + 16, &iat_rva.to_le_bytes());
        b.poke(desc_rva as usize + 12, &name_rva.to_le_bytes());
        let bytes = b.finish((desc_rva, 40), (0, 0));
        // The walk runs off the end looking for a terminator → typed error.
        assert!(matches!(parse(&bytes), Err(Error::Malformed(_))));
    }

    #[test]
    fn truncated_optional_header_is_rejected() {
        // Shrink `SizeOfOptionalHeader` below the 112-byte PE32+ fixed portion.
        // The field lives at IMAGE_FILE_HEADER + 16 (e_lfanew + 4 + 16); the
        // `< 112` guard must now reject it. Before the offset fix the parser read
        // the optional-header Magic at +20 (0x20B), so the guard never fired and
        // this image parsed as valid — the offset bug this test pins down.
        let mut bytes = Builder::new().finish((0, 0), (0, 0));
        let size_field = E_LFANEW + 4 + 16;
        bytes[size_field..size_field + 2].copy_from_slice(&100u16.to_le_bytes());
        match parse(&bytes) {
            Err(Error::Malformed(msg)) => {
                assert!(msg.contains("optional header too small"), "got: {msg}");
            }
            other => panic!("expected the truncated-optional-header guard to fire, got {other:?}"),
        }
    }

    #[test]
    fn misaligned_iat_slot_is_malformed() {
        // A crafted-but-loadable image whose FirstThunk (IAT) RVA is not
        // pointer-aligned. The Windows loader tolerates this; the engine must
        // reject it before the slot is ever reinterpreted as `&AtomicUsize`
        // (which would be UB), returning a typed error rather than faulting.
        let mut b = Builder::new();
        let desc_rva = b.rva();
        for _ in 0..10 {
            b.u32(0);
        }
        let name_rva = b.cstr("KERNEL32.dll");
        let create = b.hint_name(0, "CreateFileW");
        b.align(8);
        let int_rva = b.rva();
        b.u64(u64::from(create)); // INT[0]: name
        b.u64(0); // INT terminator
        b.align(8);
        let iat_rva = b.rva();
        b.u64(0x1111); // IAT[0] (the aligned slot lives here)
        b.u64(0); // slack so the deliberately misaligned slot stays in-bounds
        b.u64(0);
        b.poke(desc_rva as usize, &int_rva.to_le_bytes()); // OriginalFirstThunk
        b.poke(desc_rva as usize + 12, &name_rva.to_le_bytes()); // Name
        // FirstThunk offset by 4: misaligned but fully within the mapped range,
        // so it clears the bounds check and reaches the alignment guard.
        b.poke(desc_rva as usize + 16, &(iat_rva + 4).to_le_bytes());
        let bytes = b.finish((desc_rva, 40), (0, 0));
        match parse(&bytes) {
            Err(Error::Malformed(msg)) => assert!(msg.contains("aligned"), "got: {msg}"),
            other => panic!("expected a pointer-alignment rejection, got {other:?}"),
        }
    }

    /// Fuzz: neither random buffers nor bit-flipped valid images may panic or
    /// read out of bounds; every outcome must be a typed `Result`.
    #[test]
    fn fuzz_never_panics_or_reads_oob() {
        // A small xorshift PRNG keeps the test deterministic and dependency-free.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        // 1) Wholly random buffers of assorted sizes.
        for _ in 0..4000 {
            let len = (next() as usize % 4096) + 1;
            let mut buf = vec![0u8; len];
            for byte in &mut buf {
                *byte = next() as u8;
            }
            // Returns Result (Ok or Err) — the point is it must not panic.
            let _ = parse(&buf);
        }

        // 2) A valid image with random single-byte corruptions.
        let mut base = Builder::new();
        let desc_rva = base.rva();
        for _ in 0..10 {
            base.u32(0);
        }
        let name_rva = base.cstr("KERNEL32.dll");
        let create = base.hint_name(0, "CreateFileW");
        base.align(8);
        let int_rva = base.rva();
        base.u64(u64::from(create));
        base.u64(0);
        let iat_rva = base.rva();
        base.u64(0x1111);
        base.u64(0);
        base.poke(desc_rva as usize, &int_rva.to_le_bytes());
        base.poke(desc_rva as usize + 12, &name_rva.to_le_bytes());
        base.poke(desc_rva as usize + 16, &iat_rva.to_le_bytes());
        let valid = base.finish((desc_rva, 40), (0, 0));
        parse(&valid).expect("baseline valid image parses");

        for _ in 0..8000 {
            let mut corrupt = valid.clone();
            let idx = next() as usize % corrupt.len();
            corrupt[idx] ^= next() as u8;
            let _ = parse(&corrupt);
        }
    }
}
