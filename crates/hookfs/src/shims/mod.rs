//! iOS/iPadOS (identical shims). Each is gated to the exact targets whose `plthook`
//! backend can install them; other unix targets (BSD, Android/bionic) have no engine
//! yet and route through the `UnsupportedPlatform` placeholder in
//! [`crate::install`], so no shim is built.

#[cfg(windows)]
pub(crate) mod windows;

// The Unix shims are shared between Linux/glibc and Darwin (macOS + iOS): the
// portable core (the carrier-fd/read/lseek/close family, the dir shims, the
// path/mutate family, the variadic-`open` trampoline) lives in `unix`, and the
// genuinely platform-specific pieces are cfg-split — `fopencookie` vs `funopen`,
// glibc's versioned `__?xstat` + `*64` variants vs Darwin `stat`/`readdir` +
// `__getdirentries64`, the glibc vs Darwin `struct stat`/`dirent` layouts, and
// verbatim (same libc struct layouts, same `funopen`). Other unix targets (BSD,
// Android) have no engine yet.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
pub(crate) mod unix;
