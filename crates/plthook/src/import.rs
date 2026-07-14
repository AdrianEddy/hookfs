//! The format-neutral raw-import record produced by every backend parser.
//!
//! Both the PE walker ([`crate::pe`]) and the ELF walker ([`crate::elf`]) decode
//! their own on-disk / in-memory structures but describe each discovered import
//! with the same [`RawImport`] value, so the platform-agnostic slot layer
//! ([`crate::slot`]) can build an [`ImportSlot`](crate::ImportSlot) from either
//! without knowing the source format.

use crate::{ImportKind, Symbol};
use std::sync::Arc;

/// One parsed import: the providing library, the (recoverable) symbol, the
/// observed symbol version, the absolute address of the slot to patch, and how it
/// is bound.
#[derive(Debug)]
pub(crate) struct RawImport {
    /// Providing library / SONAME, shared as an `Arc<str>` so every slot of one
    /// descriptor (PE) or one `verneed` provider (ELF) references a single
    /// allocation rather than re-owning the name. Empty when the format does not
    /// record a provider for this slot (e.g. an unversioned ELF import).
    pub(crate) library: Arc<str>,
    /// `None` when the symbol name/ordinal is unrecoverable — the legacy PE
    /// `OriginalFirstThunk == 0` layout. ELF imports always carry a name.
    pub(crate) symbol: Option<Symbol>,
    /// The observed symbol version, e.g. `GLIBC_2.2.5` for an ELF import bound to
    /// versioning) and for unversioned ELF imports. Retained for diagnostics and
    /// audit cross-checks; matching is by name, so a request for `open` matches
    /// `open@GLIBC_*`.
    pub(crate) version: Option<Arc<str>>,
    /// Absolute address of the slot cell (`*mut *mut c_void`) to patch. For a live
    /// image this is the real, in-bounds, pointer-aligned patch target; for an
    /// offline (file-bytes) parse it is the relocation's link-time virtual address
    /// (used only by the host-independent parser tests, never patched).
    pub(crate) slot: usize,
    /// Standard load-time binding vs. PE delay-load. Every ELF GOT slot is
    /// [`ImportKind::Standard`] (there is no ELF delay-load; lazy PLT binding is an
    /// engine-internal detail handled by resolving originals via `dlsym`, R6).
    pub(crate) kind: ImportKind,
    /// Whether this slot holds an **authenticated** (PAC-signed) pointer — only
    /// require a correctly key/diversity/address-signed replacement that cannot be
    /// synthesized here, so the transaction refuses such a slot rather than writing
    /// a bad pointer. Always `false` for PE/ELF and for plain-arm64/x86-64 Mach-O.
    pub(crate) authenticated: bool,
}
