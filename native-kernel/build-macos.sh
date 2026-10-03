#!/usr/bin/env bash
# Cross-build the Apple Silicon (arm64) kernel from any host.
#
# No macOS machine or Xcode is needed: cargo-zigbuild drives the Rust compiler
# and uses Zig as the C compiler/linker, and Zig ships its own Darwin stubs
# rather than relying on an Apple-installed SDK. Apple Silicon supports macOS
# 11.0 (Big Sur), which is the oldest release that can run these binaries.
#
# Requires: rustup target add aarch64-apple-darwin
#           cargo install cargo-zigbuild
#           Zig >= 0.15 on PATH
set -euo pipefail

PYTHON="${PYTHON:-python3}"
# Keep the Cargo target directory out of the repo by default; some hosts
# (e.g. Windows with Application Control) refuse to execute build scripts
# written into a working tree.
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/algorithex-native-build}"
TARGET=aarch64-apple-darwin
DEPLOYMENT_TARGET=11.0
SDK=15.5
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${HERE}/../algorithex/_native/bin"

# Python symbols are supplied by the interpreter at load time, so they must be
# left undefined rather than resolved against a Python we do not have here.
export RUSTFLAGS="-C link-arg=-Wl,-undefined,dynamic_lookup"

cargo zigbuild --release --target "$TARGET" --target-dir "$BUILD_DIR"

BUILT="${BUILD_DIR}/${TARGET}/release/libalgorithex_kernel.dylib"
[ -f "$BUILT" ] || { echo "build produced no dylib" >&2; exit 1; }

# Zig stamps LC_BUILD_VERSION with its own bundled SDK default (13.0) and
# ignores -mmacosx-version-min for the dylib's load command. Rewrite that one
# field so Big Sur and Monterey can load the binary. Everything the dylib
# imports (malloc/free, libm, pthreads, dirent, dyld) predates macOS 11.
"$PYTHON" - "$BUILT" "$DEPLOYMENT_TARGET" "$SDK" <<'PY'
import struct, sys

path, minos, sdk = sys.argv[1], sys.argv[2], sys.argv[3]
encode = lambda v: int(v.split(".")[0]) << 16

with open(path, "rb") as fh:
    data = bytearray(fh.read())

ncmds = struct.unpack_from("<I", data, 16)[0]
offset = 32
for _ in range(ncmds):
    cmd, cmdsize = struct.unpack_from("<II", data, offset)
    if cmd == 0x32:  # LC_BUILD_VERSION
        platform = struct.unpack_from("<I", data, offset + 8)[0]
        assert platform == 1, f"unexpected platform {platform}, expected macOS"
        struct.pack_into("<II", data, offset + 12, encode(minos), encode(sdk))
        break
    offset += cmdsize
else:
    raise SystemExit("no LC_BUILD_VERSION load command found")

with open(path, "wb") as fh:
    fh.write(data)

version = lambda v: "%d.%d.%d" % (v >> 16, (v >> 8) & 0xFF, v & 0xFF)
print(f"deployment target set to macOS {version(encode(minos))} (sdk {version(encode(sdk))})")
PY

mkdir -p "$OUT"
cp "$BUILT" "$OUT/algorithex_kernel.dylib"
echo "vendored $OUT/algorithex_kernel.dylib"