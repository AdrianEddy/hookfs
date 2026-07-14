# hookfs

`hookfs` is a Rust workspace for presenting owned `Read + Seek` streams and
custom virtual filesystems to native, closed-source libraries as ordinary file
paths. It redirects file operations at the native module's import table, so the
data can stay in memory while unrelated paths continue to use the host file
system normally.

The workspace publishes two crates:

- [`hookfs`](crates/hookfs) is the high-level virtual-filesystem and mounting
  layer.
- [`plthook-native`](crates/plthook) is the lower-level, pure-Rust import-table
  rebinding engine. Its Rust library name remains `plthook`.

## Why hookfs?

Some native SDKs accept only a path even when an application already owns the
bytes in memory, in an archive, on a remote service, or behind an application-
specific storage abstraction. Writing a temporary file can be slow, leak
sensitive data, complicate cleanup, or simply be unavailable on a sandboxed
platform.

`hookfs` creates a private synthetic path, mounts a stream or `VirtualFs` at that
path, and patches the selected native module's imported file functions. Calls
for virtual paths are handled in Rust; calls for every other path pass through
to the original operating-system functions.

## Quick start

Add the high-level crate:

```toml
[dependencies]
hookfs = "0.1.0"
```

Then mount a stream and install hooks for the native module that will open it.
`Options::for_module` uses the supplied address only to identify that module; it
does not require a special `hookfs` export. Pass any exported function address
from the exact DLL, `.so`, or `.dylib` whose imported file calls should be
patched—normally an SDK entry point you already resolved. For example, a BRAW
integration can use the address of `CreateBlackmagicRawFactoryInstance`.

```rust
use hookfs::{Hookfs, Options};
use std::ffi::c_void;
use std::io::Cursor;
use std::path::Path;

fn main() -> hookfs::Result<()> {
    let fs = Hookfs::new();
    let mounted = fs.mount("clip.bin", Cursor::new(b"bytes in memory".to_vec()))?;

    // Use an address inside the native library whose imports should be redirected.
    let hooks = fs.install(Options::for_module(native_library_module_anchor()))?;

    call_native_library(&fs.path_for("clip.bin"));

    // Keep both guards alive while the native library may access the path.
    drop(hooks);
    drop(mounted);
    Ok(())
}

fn native_library_module_anchor() -> *const c_void {
    // Return any export address from the module to patch. For example:
    // GetProcAddress(sdk, "CreateBlackmagicRawFactoryInstance") on Windows, or
    // dlsym(sdk, "CreateBlackmagicRawFactoryInstance") on Unix.
    todo!("return an export address from the target native module")
}

fn call_native_library(_path: &Path) {
    // Pass the synthetic path to the native SDK.
}
```

See the [`hookfs` crate README](crates/hookfs/README.md) for custom providers,
writes, module scopes, late-loaded modules, and lifecycle details. Use
`plthook-native` directly when you need general import rebinding rather than file
virtualization; its API is documented in the
[`plthook-native` crate README](crates/plthook/README.md).

## Supported targets

The live backends currently support 64-bit `x86_64` and `aarch64` targets on:

| Operating system | Object format | Rebinding table |
| --- | --- | --- |
| Windows | PE32+ | IAT and delay-import IAT |
| Linux (glibc) | ELF64 | PLT/GOT relocations |
| macOS | Mach-O 64 | symbol pointers and chained fixups |
| iOS/iPadOS | Mach-O 64 | symbol pointers and chained fixups |

Authenticated arm64e pointer slots are detected and rejected because writing an
unsigned replacement would be unsafe. Other operating systems compile the
platform-independent `hookfs` API, but installing hooks returns
`UnsupportedPlatform`.

## Safety model and limitations

Import-table rebinding affects only calls that actually pass through a patched
slot in a selected, already-loaded module. Inlined calls, direct syscalls,
statically linked implementations, and calls made by unpatched modules are not
intercepted. `hookfs` is therefore intended for controlled native-library
integration, not for process isolation or as a security boundary.

Install and restore operations are serialized. Replacement slots are written
atomically, page protections are restored, installation rolls back on failure,
and RAII guards restore slots only when they still contain the value owned by
that guard. Application code must still ensure the target module remains loaded
and that no native call outlives the relevant mount or installation guard.

## Development

The repository uses Rust 2024 and requires Rust 1.96 or newer.

```console
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

Platform integration tests compile small native-call fixtures from
`crates/*/tests/fixtures`. Run them natively on each supported operating system.

For release validation, package the low-level dependency before `hookfs`:

```console
cargo package -p plthook-native
cargo package -p hookfs
```

Publish in the same order. After `plthook-native` is available on crates.io, publish
`hookfs`, whose manifest uses the registry version while retaining a path for
workspace development.

## Licensing and attribution

Original work in this repository is available under either the MIT License or
the Apache License 2.0, at your option.

`plthook-native` is a Rust port derived from
[`kubo/plthook`](https://github.com/kubo/plthook), with upstream notices spanning
2013-2024 for its platform implementations, and includes that project's
BSD-2-Clause terms. See the license and notice files distributed with each crate
for the exact terms.
