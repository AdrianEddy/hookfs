//! Typed error model for `hookfs` installation and mounting.
//!
//! Runtime shim failures never surface as a Rust `Error`: they are translated
//! directly — mounting streams and installing hooks.

/// Errors produced while mounting streams or installing the filesystem hooks.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The underlying import-rebinding engine failed to acquire, parse, or patch
    /// the target module(s).
    #[error("import-hook engine error: {0}")]
    Engine(#[from] plthook::Error),

    /// The scope resolved to no target module — e.g. an address that belongs to
    /// no loaded image, or a name filter that matched nothing.
    #[error("no target module found for the requested scope: {0}")]
    NoTargetModule(&'static str),

    /// An installation is already active. `hookfs` enforces a **single active
    /// routing engine backs every shim, so a second [`install`](crate::install)
    /// while an [`InstallGuard`](crate::InstallGuard) is still live fails loudly
    /// here instead of silently clobbering the live installation. Drop the existing
    /// guard before installing again.
    #[error(
        "a hookfs installation is already active in this process (single active installation only)"
    )]
    AlreadyInstalled,

    /// A logical mount name was not a valid single path component (empty, absolute,
    /// contained a separator into a parent, an embedded NUL, or a `..` escape).
    #[error("invalid mount name `{name}`: {reason}")]
    InvalidMountName {
        /// The offending logical name.
        name: String,
        /// Why it was rejected.
        reason: &'static str,
    },

    /// A backing stream could not report its size while mounting (size is captured
    /// once, up front, so immutable mounts answer `stat`/`GetFileSizeEx` without a
    /// per-call seek dance).
    #[error("could not determine the size of the stream for `{name}`: {source}")]
    StreamSize {
        /// The logical name being mounted.
        name: String,
        /// The I/O error from the size probe.
        source: std::io::Error,
    },

    #[error("hookfs shims are unavailable on this platform")]
    UnsupportedPlatform,
}

/// Crate-local result alias.
pub type Result<T> = core::result::Result<T, Error>;
