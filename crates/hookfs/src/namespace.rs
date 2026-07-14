//! The reserved synthetic namespace and its lexical path matching
//!
//! A mount lives under a reserved prefix that is syntactically an ordinary path
//! but never resolves to a real user file:
//!
//! ```text
//! Windows: C:\__hookfs__\<mount-id>\clip.braw
//! POSIX:   /__hookfs__/<mount-id>/clip.braw
//! ```
//!
//! `<mount-id>` is process-random so a synthetic path cannot collide with a real
//! file (R18). Membership is decided **lexically, before any physical
//! canonicalization** — case-insensitively and separator-normalized on Windows,
//! byte-preserving and case-sensitive on POSIX — and `..` escapes, embedded NULs,
//! and device/stream tricks are rejected. A virtual path therefore never falls

use std::path::{Path, PathBuf};

/// The fixed first component of every reserved path. Chosen to be visibly
/// synthetic and unlikely to exist on a real volume.
const RESERVED_ROOT: &str = "__hookfs__";

/// The drive the reserved tree hangs under (Windows). `C:` is the system volume,
/// always present, so ordinary path/volume helpers (`GetFullPathNameW`,
/// `GetDriveTypeW`) give sane real answers for the *drive* while the *subtree*
/// stays virtual. POSIX roots the tree at `/` instead (no drive concept).
#[cfg(windows)]
const RESERVED_DRIVE: &str = "C:";

/// The native path separator used to build and prefix-match reserved paths, and
/// (in [`normalize_key`]) the separator every normalized key is expressed with —
/// which is why [`crate::providers::memory`] reuses it for its key arithmetic
/// rather than re-deriving it.
#[cfg(windows)]
pub(crate) const SEP: char = '\\';
#[cfg(unix)]
pub(crate) const SEP: char = '/';

/// The reserved namespace for one mount context: a random subtree root plus the
/// predicate that decides membership.
pub(crate) struct Namespace {
    /// The virtual directory, e.g. `C:\__hookfs__\<id>` (display form, for Debug).
    root: PathBuf,
    /// The membership predicate. The default matches the reserved prefix; callers
    /// may override it (`Options::is_virtual` / `Options::for_prefix`).
    predicate: Box<dyn Fn(&Path) -> bool + Send + Sync>,
}

impl std::fmt::Debug for Namespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Namespace")
            .field("root", &self.root)
            .finish_non_exhaustive()
    }
}

/// A fresh, process-random reserved subtree root (`C:\__hookfs__\<id>` on Windows,
/// `/__hookfs__/<id>` on POSIX). The convenience mount layer and
/// [`Namespace::reserved`] share this so a mount's synthetic paths match the
/// installed namespace.
#[cfg(windows)]
pub(crate) fn reserved_root() -> PathBuf {
    PathBuf::from(format!(
        "{RESERVED_DRIVE}{SEP}{RESERVED_ROOT}{SEP}{}",
        random_mount_id()
    ))
}

/// POSIX form: `/__hookfs__/<id>`.
#[cfg(unix)]
pub(crate) fn reserved_root() -> PathBuf {
    PathBuf::from(format!("{SEP}{RESERVED_ROOT}{SEP}{}", random_mount_id()))
}

impl Namespace {
    /// A prefix namespace rooted at `root` (its whole subtree is virtual).
    pub(crate) fn from_root(root: PathBuf) -> Self {
        let key_for_predicate = normalize_key_str(&root.to_string_lossy());
        let predicate =
            Box::new(move |path: &Path| key_has_prefix(&normalize_key(path), &key_for_predicate));
        Self { root, predicate }
    }

    /// A namespace rooted at a fresh, process-random reserved subtree.
    pub(crate) fn reserved() -> Self {
        Self::from_root(reserved_root())
    }

    /// A namespace matching an arbitrary caller-chosen prefix (`Options::for_prefix`).
    pub(crate) fn for_prefix(prefix: impl Into<PathBuf>) -> Self {
        Self::from_root(prefix.into())
    }

    /// A namespace with a fully custom membership predicate but the reserved root
    /// (used when a caller supplies `Options::is_virtual`).
    pub(crate) fn with_predicate(predicate: Box<dyn Fn(&Path) -> bool + Send + Sync>) -> Self {
        let mut ns = Self::reserved();
        ns.predicate = predicate;
        ns
    }

    /// Whether `path` is inside this namespace, per the active predicate. `path`
    /// must already be lexically normalized/absolute for a reliable answer;
    /// [`Router`](crate::router) does that first. Only the routing engine consults
    /// this, so it exists only where a shim backend does (`hookfs_backend`).
    #[cfg(hookfs_backend)]
    pub(crate) fn is_virtual(&self, path: &Path) -> bool {
        (self.predicate)(path)
    }
}

/// Reject strings that must never be treated as a benign virtual path: an
/// embedded NUL, a `..`-style traversal component, or an alternate-data-stream /
/// device marker. Returns `false` (unsafe) so the caller fails closed.
#[cfg(windows)]
pub(crate) fn is_lexically_safe(decoded: &str) -> bool {
    if decoded.contains('\0') {
        return false;
    }
    for raw in decoded.split(['\\', '/']) {
        // Reject any dot/space-only navigation segment (`.`, `..`, and Windows'
        // trailing-dot/space variants like `.. `, `...`, `. .`). Empirically only
        // the exact `..` traverses on real KERNEL32 — `CreateFileW` on `sub\.. \x`
        // yields `ERROR_PATH_NOT_FOUND`, it does not escape to the parent — but no
        // such component is ever a legitimate mounted name, so failing closed on
        // all of them is correct defense-in-depth and never over-rejects a real
        // file name (see `is_dot_component`).
        if is_dot_component(raw) {
            return false;
        }
        // A colon anywhere other than the drive designator (`C:`) is a stream or
        // device marker on Windows. Windows strips trailing dots and spaces from a
        // component before it reaches the filesystem, so compare the trimmed form.
        let comp = raw.trim_end_matches([' ', '.']);
        if comp.contains(':')
            && !(comp.len() == 2 && comp.as_bytes().first().is_some_and(u8::is_ascii_alphabetic))
        {
            return false;
        }
    }
    true
}

/// POSIX form: reject an embedded NUL and any `.`/`..` traversal component. Unlike
/// Windows there are no drive/stream/device markers, and a colon is a legal file
#[cfg(unix)]
pub(crate) fn is_lexically_safe(decoded: &str) -> bool {
    if decoded.contains('\0') {
        return false;
    }
    // Byte-preserving, case-sensitive: split on `/` only (a backslash is an
    // ordinary file-name byte on POSIX, never a separator).
    !decoded.split('/').any(|comp| comp == "." || comp == "..")
}

/// Whether a path component is a pure dot/space navigation segment: made up solely
/// of `.` and space characters and containing at least one `.` (`.`, `..`, `...`,
/// `.. `, ` ..`, `. .`, …). Windows strips trailing dots and spaces from a
/// component, so every one of these normalizes to `.`/`..` or a degenerate empty
/// name and is never a legitimate file name. A component that merely *contains*
/// dots (`clip.braw`, `..foo`, `a...b`) or is all spaces is left alone.
#[cfg(windows)]
fn is_dot_component(comp: &str) -> bool {
    !comp.is_empty()
        && comp.bytes().all(|b| b == b'.' || b == b' ')
        && comp.bytes().any(|b| b == b'.')
}

/// Normalize a decoded path string into its canonical comparison key: strip a
/// `\\?\` (or `\\?\UNC\`) long-path prefix, unify separators to `\`, collapse
/// repeats, drop a trailing separator, and lowercase (ASCII) — the Windows
#[cfg(windows)]
pub(crate) fn normalize_key_str(input: &str) -> String {
    let mut s = input;
    // Strip long-path prefixes.
    if let Some(rest) = s.strip_prefix("\\\\?\\UNC\\") {
        // Represent the UNC body with a leading `\\` so distinct shares stay distinct.
        return normalize_key_str(&format!("\\\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix("\\\\?\\") {
        s = rest;
    }
    let mut out = String::with_capacity(s.len());
    let mut prev_sep = false;
    for ch in s.chars() {
        let c = if ch == '/' { '\\' } else { ch };
        if c == '\\' {
            if prev_sep {
                continue; // collapse `\\` runs (but keep a single leading one)
            }
            prev_sep = true;
            out.push('\\');
        } else {
            prev_sep = false;
            out.extend(c.to_lowercase());
        }
    }
    // Drop a single trailing separator (but never reduce a bare root like `c:\`).
    if out.len() > 3 && out.ends_with('\\') {
        out.pop();
    }
    out
}

/// POSIX form: byte-preserving and **case-sensitive** — collapse repeated `/`,
/// drop a single trailing `/` (but never the bare root `/`), and leave the bytes
/// otherwise untouched. A backslash is an ordinary file-name byte, not a
#[cfg(unix)]
pub(crate) fn normalize_key_str(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut prev_sep = false;
    for c in input.chars() {
        if c == '/' {
            if prev_sep {
                continue; // collapse `//` runs (but keep a single leading one)
            }
            prev_sep = true;
            out.push('/');
        } else {
            prev_sep = false;
            out.push(c);
        }
    }
    if out.len() > 1 && out.ends_with('/') {
        out.pop();
    }
    out
}

/// [`normalize_key_str`] for a `Path`.
pub(crate) fn normalize_key(path: &Path) -> String {
    normalize_key_str(&path.to_string_lossy())
}

/// Whether normalized `key` is `prefix` itself or a descendant of it (component
/// boundary honored via the native separator).
fn key_has_prefix(key: &str, prefix: &str) -> bool {
    if let Some(rest) = key.strip_prefix(prefix) {
        rest.is_empty() || rest.starts_with(SEP)
    } else {
        false
    }
}

/// Decode a NUL-terminated UTF-16 pointer into a `Vec<u16>` (without the NUL),
/// scanning at most `MAX` code units so a malformed pointer cannot run away.
///
/// # Safety
/// `ptr` must be null or point to a readable, NUL-terminated UTF-16 string.
#[cfg(windows)]
pub(crate) unsafe fn decode_wide(ptr: *const u16) -> Option<Vec<u16>> {
    const MAX: usize = 0x8000; // 32K code units — far beyond any real path.
    if ptr.is_null() {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..MAX {
        // SAFETY: caller guarantees a readable NUL-terminated buffer; we stop at
        // the terminator and never read past `MAX`.
        let unit = unsafe { *ptr.add(i) };
        if unit == 0 {
            return Some(out);
        }
        out.push(unit);
    }
    None
}

/// Decode a NUL-terminated ANSI/UTF-8-ish byte pointer into a `String` (lossy),
/// scanning at most `MAX` bytes.
///
/// # Safety
/// `ptr` must be null or point to a readable, NUL-terminated byte string.
#[cfg(windows)]
pub(crate) unsafe fn decode_ansi(ptr: *const u8) -> Option<String> {
    const MAX: usize = 0x8000;
    if ptr.is_null() {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..MAX {
        // SAFETY: caller guarantees a readable NUL-terminated buffer.
        let byte = unsafe { *ptr.add(i) };
        if byte == 0 {
            return Some(String::from_utf8_lossy(&out).into_owned());
        }
        out.push(byte);
    }
    None
}

/// Decode a NUL-terminated C byte string (`*const c_char`) into a `String`
/// (lossy), scanning at most `MAX` bytes so a malformed pointer cannot run away.
/// The POSIX shim path decoder — POSIX file names are opaque bytes, decoded
/// lossily for matching while the original bytes stay in `Path`.
///
/// # Safety
/// `ptr` must be null or point to a readable, NUL-terminated byte string.
///
/// Only the unix shims decode paths this way, so it exists only where a unix shim
#[cfg(all(unix, hookfs_backend))]
pub(crate) unsafe fn decode_cstr(ptr: *const core::ffi::c_char) -> Option<String> {
    const MAX: usize = 0x8000;
    if ptr.is_null() {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..MAX {
        // SAFETY: caller guarantees a readable NUL-terminated buffer; we stop at
        // the terminator and never read past `MAX`. `c_char` is `i8` on Linux;
        // reinterpret its bit pattern as the raw byte.
        let byte = u8::from_ne_bytes(unsafe { *ptr.add(i) }.to_ne_bytes());
        if byte == 0 {
            return Some(String::from_utf8_lossy(&out).into_owned());
        }
        out.push(byte);
    }
    None
}

/// Case-insensitive DOS wildcard match for directory enumeration (`*` = any run,
/// `?` = any single char). Sufficient for the exact-name and `dir\*` / `*.braw`
/// patterns the SDK uses; `<` `>` `"` DOS-quirk wildcards are not needed here.
/// Windows-only: POSIX directory reads (`readdir`) enumerate every entry, they do
/// not glob.
#[cfg(windows)]
pub(crate) fn wildcard_match(pattern: &str, name: &str) -> bool {
    let pat: Vec<char> = pattern.chars().flat_map(char::to_lowercase).collect();
    let txt: Vec<char> = name.chars().flat_map(char::to_lowercase).collect();
    matches_from(&pat, &txt)
}

#[cfg(windows)]
fn matches_from(pat: &[char], txt: &[char]) -> bool {
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star_pat, mut star_txt): (Option<usize>, usize) = (None, 0);
    while ti < txt.len() {
        match pat.get(pi) {
            Some('*') => {
                star_pat = Some(pi);
                star_txt = ti;
                pi += 1;
            }
            Some('?') => {
                pi += 1;
                ti += 1;
            }
            Some(&c) if txt.get(ti) == Some(&c) => {
                pi += 1;
                ti += 1;
            }
            _ => {
                if let Some(sp) = star_pat {
                    pi = sp + 1;
                    star_txt += 1;
                    ti = star_txt;
                } else {
                    return false;
                }
            }
        }
    }
    while pat.get(pi) == Some(&'*') {
        pi += 1;
    }
    pi == pat.len()
}

/// A process-random 128-bit mount id, hex-encoded. Uses the OS-seeded
/// `RandomState` plus the pid/time so two processes (and two mounts) never share
/// a reserved subtree.
fn random_mount_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let state = std::collections::hash_map::RandomState::new();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX));
    let mut h1 = state.build_hasher();
    h1.write_u64(u64::from(std::process::id()));
    h1.write_u64(nanos);
    let a = h1.finish();
    let mut h2 = state.build_hasher();
    h2.write_u64(a);
    h2.write_usize(&raw const state as usize);
    let b = h2.finish();
    format!("{a:016x}{b:016x}")
}

#[cfg(all(test, windows))]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_separators_case_and_long_prefix() {
        assert_eq!(normalize_key_str("C:/Foo/Bar/"), "c:\\foo\\bar");
        assert_eq!(normalize_key_str("\\\\?\\C:\\Foo\\BAR"), "c:\\foo\\bar");
        assert_eq!(normalize_key_str("C:\\a\\\\b"), "c:\\a\\b");
        assert_eq!(normalize_key_str("c:\\"), "c:\\");
    }

    #[test]
    fn prefix_matching_respects_component_boundary() {
        let prefix = "c:\\__hookfs__\\abc";
        assert!(key_has_prefix("c:\\__hookfs__\\abc", prefix));
        assert!(key_has_prefix("c:\\__hookfs__\\abc\\clip.braw", prefix));
        assert!(!key_has_prefix("c:\\__hookfs__\\abcdef\\clip.braw", prefix));
        assert!(!key_has_prefix("c:\\other", prefix));
    }

    #[test]
    fn rejects_traversal_and_streams() {
        assert!(is_lexically_safe("C:\\__hookfs__\\id\\clip.braw"));
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\..\\..\\secret"));
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\clip.braw:stream"));
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\clip\0.braw"));
    }

    #[test]
    fn rejects_trailing_dot_and_space_traversal_variants() {
        // The exact `..` and its Windows trailing-dot/space variants are rejected.
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\.. \\secret")); // ".. "
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\...\\secret")); // "..."
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\. \\clip")); // ". "
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\.\\clip")); // "."
        assert!(!is_lexically_safe("C:\\__hookfs__\\id\\..  \\x")); // "..  "
        // Mixed separators are split identically.
        assert!(!is_lexically_safe("C:/__hookfs__/id/.. /secret"));
        assert!(!is_lexically_safe("C:\\__hookfs__/id\\..\\x"));
        // Legitimate names that merely contain dots (incl. leading dots) still pass.
        assert!(is_lexically_safe("C:\\__hookfs__\\id\\clip.braw"));
        assert!(is_lexically_safe("C:\\__hookfs__\\id\\..foo")); // a real name
        assert!(is_lexically_safe("C:\\__hookfs__\\id\\a...b"));
        assert!(is_lexically_safe("C:\\__hookfs__\\id\\file.name.ext"));
    }

    #[test]
    fn is_dot_component_classification() {
        for bad in [".", "..", "...", ".. ", " ..", ". .", "..  ", "   .   "] {
            assert!(is_dot_component(bad), "{bad:?} should be a dot component");
        }
        for good in ["", "clip.braw", "..foo", "a..", "foo.", "   ", "..a"] {
            assert!(
                !is_dot_component(good),
                "{good:?} should not be a dot component"
            );
        }
    }

    #[test]
    fn wildcards() {
        assert!(wildcard_match("*", "anything.braw"));
        assert!(wildcard_match("*.braw", "SAMPLE.BRAW"));
        assert!(wildcard_match("sample.braw", "sample.braw"));
        assert!(!wildcard_match("*.sidecar", "sample.braw"));
        assert!(wildcard_match("sam???.braw", "sample.braw"));
    }

    #[test]
    fn namespace_matches_children_only() {
        let ns = Namespace::from_root(PathBuf::from("C:\\__hookfs__\\abc"));
        assert!(ns.is_virtual(Path::new("C:\\__hookfs__\\abc\\clip.braw")));
        assert!(ns.is_virtual(Path::new("c:/__HOOKFS__/abc")));
        assert!(!ns.is_virtual(Path::new("C:\\__hookfs__\\abcdef\\clip.braw")));
        assert!(!ns.is_virtual(Path::new("C:\\Windows\\notepad.exe")));
    }

    #[test]
    fn reserved_roots_are_process_random() {
        assert_ne!(reserved_root(), reserved_root());
    }
}

#[cfg(all(test, unix))]
#[allow(clippy::unwrap_used)]
mod posix_tests {
    use super::*;

    #[test]
    fn normalizes_case_sensitively_and_collapses_separators() {
        // Byte-preserving + case-sensitive: no lowercasing, `/` separators.
        assert_eq!(normalize_key_str("/Foo/Bar/"), "/Foo/Bar");
        assert_eq!(normalize_key_str("/a//b"), "/a/b");
        assert_eq!(normalize_key_str("/"), "/");
        // A backslash is an ordinary file-name byte, never a separator.
        assert_eq!(normalize_key_str("/a\\b"), "/a\\b");
    }

    #[test]
    fn prefix_matching_respects_component_boundary() {
        let prefix = "/__hookfs__/abc";
        assert!(key_has_prefix("/__hookfs__/abc", prefix));
        assert!(key_has_prefix("/__hookfs__/abc/clip.braw", prefix));
        assert!(!key_has_prefix("/__hookfs__/abcdef/clip.braw", prefix));
        assert!(!key_has_prefix("/other", prefix));
    }

    #[test]
    fn rejects_traversal_and_nul_but_allows_colons() {
        assert!(is_lexically_safe("/__hookfs__/id/clip.braw"));
        assert!(!is_lexically_safe("/__hookfs__/id/../../secret"));
        assert!(!is_lexically_safe("/__hookfs__/id/./clip"));
        assert!(!is_lexically_safe("/__hookfs__/id/clip\0.braw"));
        // A colon is a legal POSIX file-name byte (unlike Windows streams).
        assert!(is_lexically_safe("/__hookfs__/id/a:b.braw"));
        // Names that merely contain dots are fine.
        assert!(is_lexically_safe("/__hookfs__/id/..foo"));
        assert!(is_lexically_safe("/__hookfs__/id/a...b"));
    }

    #[cfg(hookfs_backend)]
    #[test]
    fn namespace_matches_children_only_case_sensitively() {
        let ns = Namespace::from_root(PathBuf::from("/__hookfs__/abc"));
        assert!(ns.is_virtual(Path::new("/__hookfs__/abc/clip.braw")));
        assert!(ns.is_virtual(Path::new("/__hookfs__/abc")));
        assert!(!ns.is_virtual(Path::new("/__hookfs__/abcdef/clip.braw")));
        // Case-sensitive: a different case is a different (real) path.
        assert!(!ns.is_virtual(Path::new("/__HOOKFS__/abc")));
        assert!(!ns.is_virtual(Path::new("/usr/bin/ls")));
    }

    #[test]
    fn reserved_root_is_posix_and_random() {
        let root = reserved_root();
        assert!(root.to_string_lossy().starts_with("/__hookfs__/"));
        assert_ne!(reserved_root(), reserved_root());
    }
}
