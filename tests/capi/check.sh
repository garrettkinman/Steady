#!/usr/bin/env bash
# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# C packaging verification.
#
# Builds a generated model as a static library with --app:staticlib, then
# compiles a C program that knows nothing but the emitted header and links
# against the archive with a plain C toolchain. Three separate claims:
#
#   1. the library builds under --os:any --mm:none, so it carries no runtime
#   2. the header is self-sufficient — the C file includes nothing else
#   3. the packaged library computes what the Nim module computes, which is
#      checked by diffing its output against the same run through Nim
#
# (3) is the one that would otherwise be taken on faith. "It linked" and "it
# still gives the right answers" are different claims.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$ROOT/build/capi"
GEN="$ROOT/tests/generated"
LIB="$BUILD/libtiny_cnn.a"

if [ ! -f "$GEN/tiny_cnn_api.nim" ]; then
  echo "generated C API missing; run 'nimble gen' first" >&2
  exit 1
fi

rm -rf "$BUILD"
mkdir -p "$BUILD"

echo "==> Building a static library (--app:staticlib --os:any --mm:none)"
nim c --hints:off \
      --app:staticlib --os:any --mm:none --panics:on --noMain \
      -d:danger -d:useMalloc \
      --path:"$ROOT/src" --path:"$ROOT/tests/freestanding" \
      --nimcache:"$BUILD/nimcache" --out:"$LIB" \
      "$GEN/tiny_cnn_api.nim"
echo "    $(basename "$LIB") $(stat -c%s "$LIB") bytes"

echo "==> Symbols the archive exports"
# Only the documented entry points should be global text symbols with our
# prefix; a typo in an exportc name would otherwise go unnoticed until a
# customer's link failed.
for sym in init invoke arena_size input0 output0; do
  if ! nm "$LIB" | grep -qE " T steady_tiny_cnn_$sym$"; then
    echo "FAIL: steady_tiny_cnn_$sym is not exported"
    exit 1
  fi
  echo "    steady_tiny_cnn_$sym"
done

echo "==> Compiling a C consumer against the header alone"
gcc -std=c99 -Wall -Wextra -Werror -O2 \
    -I"$GEN" -o "$BUILD/consumer" "$ROOT/tests/capi/main.c" "$LIB"
echo "    ok"

echo "==> Running it"
"$BUILD/consumer" > "$BUILD/c_output.txt"
if grep -q FAIL "$BUILD/c_output.txt"; then
  cat "$BUILD/c_output.txt"
  exit 1
fi

echo "==> Comparing against the same run through Nim"
nim c -r --hints:off --path:"$ROOT/src" --out:"$BUILD/expected" \
    "$ROOT/tests/capi/expected.nim" > "$BUILD/nim_output.txt"
if ! diff -u "$BUILD/nim_output.txt" "$BUILD/c_output.txt"; then
  echo "FAIL: the packaged library disagrees with the Nim module"
  exit 1
fi
sed 's/^/    /' "$BUILD/c_output.txt"

echo
echo "C packaging check passed"
