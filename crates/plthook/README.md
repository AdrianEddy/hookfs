# plthook-native

`plthook-native` is a pure-Rust engine for enumerating and transactionally replacing
imports in loaded native modules. It supports PE import address tables, ELF
PLT/GOT relocations, and Mach-O symbol pointers and chained fixups.

The crates.io package is named `plthook-native` to distinguish it from the existing
`plthook` package. The Rust library name intentionally remains `plthook`, so code
imports it as `use plthook::...`.

This crate is the low-level engine used by `hookfs`. Use it directly for
instrumentation, compatibility shims, controlled native-library interposition,
or other cases where a specific loaded module's imports must be rebound.

## Installation

```toml
[dependencies]
plthook-native = "0.1.0"
```

```rust
use plthook::{Module, Replacement, Symbol};
```

The minimum supported Rust version is 1.96.

## Basic workflow

Acquire a currently loaded module, describe one or more replacement pointers,
and keep the returned guard alive for as long as calls should be redirected.
`Module::from_address` accepts any code or data address inside the exact loaded
module to patch; an address of an export already resolved by the application is
usually the most convenient anchor:

```rust
use core::ffi::c_void;
use plthook::{install, Module, Replacement};

extern "C" fn replacement() -> i32 {
    42
}

fn main() -> plthook::Result<()> {
    // SAFETY: this address belongs to a module that remains loaded for the
    // guard's entire lifetime.
    let module = unsafe { Module::from_address(target_module_anchor())? };

    let guard = install(
        &module,
        &[Replacement::by_symbol(
            "function_imported_by_target",
            replacement as *const c_void,
        )],
    )?;

    // Calls made through the target module's matched import now reach replacement.
    drop(guard); // restores slots still owned by this guard
    Ok(())
}

fn target_module_anchor() -> *const c_void {
    // Return any address inside the exact module whose imports will be patched.
    // The address identifies the module; plthook does not call this function.
    todo!("return an export address from the target module")
}
```

The replacement function must exactly match the imported function's ABI and
signature. Casting an incompatible function or allowing the target module to
unload while its `Module`/`HookGuard` is in use can violate memory safety; these
contracts are necessarily the caller's responsibility.

## Module acquisition and enumeration

`Module` represents a validated image that is already loaded in the current
process. It can be acquired from:

- `Module::from_address` for a live code or data address in the image;
- `Module::from_handle` for a live platform loader handle;
- `Module::from_name` without intentionally loading a new image; or
- `Module::enumerate` to inspect all supported loaded images.

`Module::imports` returns `ImportSlot` values with the provider name, symbol,
slot address, import kind, version where available, and protection metadata.

## Replacement matching

`Replacement::by_name(library, symbol, pointer)` restricts a match to one
provider. This is normally preferred on Windows, where provider DLL names are
matched case-insensitively. `Replacement::by_symbol(symbol, pointer)` matches a
symbol regardless of provider and is useful on ELF and Mach-O. Append
`.optional()` when absence of a symbol should not abort the transaction.

By default a requested replacement is required. Installation resolves every
target and every canonical original before changing memory. If any required
symbol or original cannot be resolved, no slot is modified.

`HookGuard::installed` describes the slots that were changed, and
`HookGuard::original` exposes the canonical original pointer for a symbol.
Calling `uninstall` restores synchronously and reports conflicts; dropping the
guard performs best-effort restoration.

## Transaction and concurrency behavior

- All install/uninstall operations are process-wide serialized.
- Protection changes are page-batched and restored to their prior values.
- Aligned pointer-width slot updates are atomic on supported architectures.
- A failed multi-symbol install rolls back already-written slots.
- Restoration uses compare-exchange and does not overwrite another component's
  later modification of the same slot.
- A disappeared module is not dereferenced during restoration.

These properties protect the patching transaction, but they cannot make an
arbitrary replacement function ABI-safe or coordinate unrelated hooking
libraries automatically.

## Supported targets

| Operating system | Format and imports | Architectures |
| --- | --- | --- |
| Windows | PE32+ standard and delay IAT | x86-64, aarch64 |
| Linux (glibc) | ELF64 `JUMP_SLOT` and `GLOB_DAT` | x86-64, aarch64 |
| macOS | Mach-O symbol pointers/chained fixups | x86-64, arm64 |
| iOS/iPadOS | Mach-O symbol pointers/chained fixups | arm64 |

Only calls emitted through a patched import slot are affected. Direct calls,
inlined code, direct syscalls, and static linkage are outside this mechanism.
Authenticated arm64e PAC-signed slots are detected and rejected; writing an
unsigned pointer to such a slot would be invalid.

## Origin, license, and attribution

`plthook-native` is a Rust port derived from
[`kubo/plthook`](https://github.com/kubo/plthook). The upstream implementation is
covered by Kubo Takehiro's platform-specific 2013-2024 copyright notices and
distributed under the BSD-2-Clause license. Its notices and disclaimer are
retained in `LICENSE-BSD-2-Clause` and `NOTICE`.

Original Rust work is available under either MIT or Apache-2.0, at your option.
Because the crate incorporates ported BSD-2-Clause work, redistribution must
also comply with the retained BSD notice. The package manifest expresses this
as `(MIT OR Apache-2.0) AND BSD-2-Clause`.
