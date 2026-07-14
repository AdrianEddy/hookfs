//! Platform-agnostic value types shared by the format parsers and the patching
//! engine. Keeping `Symbol`/`ImportKind` here (rather than in the Windows-only
//! [`crate::slot`] module) lets the pure parser in [`crate::pe`] — and the ELF /
//! platform code.

/// An imported symbol, identified either by name or by ordinal.
///
/// Function names are matched **exactly** (case-sensitive) — correct for 64-bit
/// providing library is matched separately and case-insensitively.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Symbol {
    /// Import by name (e.g. `CreateFileW`).
    Name(String),
    /// Import by ordinal (e.g. `OLEAUT32.dll` `#8`).
    Ordinal(u16),
}

impl Symbol {
    /// Borrow a name symbol from a string slice.
    #[must_use]
    pub fn name(name: &str) -> Self {
        Self::Name(name.to_owned())
    }

    /// Whether two symbols denote the same import (name equality or ordinal
    /// equality). A name never matches an ordinal.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Name(a), Self::Name(b)) => a == b,
            (Self::Ordinal(a), Self::Ordinal(b)) => a == b,
            _ => false,
        }
    }

    /// Human-readable form for diagnostics (`CreateFileW`, `#8`).
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Name(name) => name.clone(),
            Self::Ordinal(ordinal) => format!("#{ordinal}"),
        }
    }
}

impl core::fmt::Display for Symbol {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Name(name) => f.write_str(name),
            Self::Ordinal(ordinal) => write!(f, "#{ordinal}"),
        }
    }
}

/// How an import reaches its slot: a normal load-time binding or a delay-load
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportKind {
    /// `IMAGE_DIRECTORY_ENTRY_IMPORT`.
    Standard,
    /// `IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT`.
    DelayLoad,
}
