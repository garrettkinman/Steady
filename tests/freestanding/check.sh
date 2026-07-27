#!/usr/bin/env bash
# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Bare-metal build verification.
#
# Two claims are checked, because they are different claims:
#   1. the runtime and a generated model compile under --os:any --mm:none
#   2. nothing in them actually pulls in an allocator
#
# (2) is the one worth automating. Nim will happily compile code that links
# malloc from a path you never call, and "no dynamic allocation" is not a
# property you want to be taking on faith.
#
# If arm-none-eabi-gcc is present the check cross-compiles for Cortex-M4 and
# inspects a fully linked image. Otherwise it stops after Nim codegen.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$ROOT/build/freestanding"
NIMLIB="$(nim --hints:off --eval:'import std/os; echo getCurrentCompilerExe().parentDir.parentDir / "lib"' 2>/dev/null | tail -1)"
if [ ! -f "$NIMLIB/nimbase.h" ]; then
  echo "could not locate the Nim lib directory (looked in '$NIMLIB')" >&2
  exit 1
fi

rm -rf "$BUILD"

echo "==> Nim codegen (--os:any --cpu:arm --mm:none --panics:on)"
nim c --hints:off \
      --path:"$ROOT/src" --path:"$ROOT/tests/freestanding" \
      --os:any --cpu:arm --mm:none --panics:on --noMain \
      --compileOnly --nimcache:"$BUILD" \
      -d:danger -d:useMalloc \
      "$ROOT/tests/freestanding/main.nim"
echo "    ok"

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "==> arm-none-eabi-gcc not found; skipping the link audit"
  exit 0
fi

echo "==> Cross-compiling for Cortex-M4"
cd "$BUILD"
CFLAGS="-c -Os -ffreestanding -ffunction-sections -fdata-sections -mcpu=cortex-m4 -mthumb -w -I$NIMLIB"
for f in *.c; do arm-none-eabi-gcc $CFLAGS -o "${f%.c}.o" "$f"; done
arm-none-eabi-gcc $CFLAGS -o weights.o "$ROOT/tests/generated/tiny_cnn_weights.c"
arm-none-eabi-gcc $CFLAGS -o weights_branch.o "$ROOT/tests/generated/branch_net_weights.c"

cat > root.c <<'EOF'
/* Roots for --gc-sections: without these the linker would discard the
   entire library and the audit below would prove nothing. */
extern int steady_selftest(void);
extern void steady_model_invoke(void);
extern signed char *steady_model_input(void);
extern signed char *steady_model_output(void);
extern int steady_arena_size(void);
extern void steady_branch_invoke(void);
extern signed char *steady_branch_input(void);
extern signed char *steady_branch_output(void);
extern int steady_branch_arena_size(void);
void _start(void) {
  steady_selftest(); steady_model_invoke();
  steady_model_input(); steady_model_output(); steady_arena_size();
  steady_branch_invoke();
  steady_branch_input(); steady_branch_output(); steady_branch_arena_size();
  for (;;) { }
}
EOF
arm-none-eabi-gcc $CFLAGS -o root.o root.c

arm-none-eabi-gcc -o image.elf root.o ./@m*.o ./@p*.o weights.o weights_branch.o \
  -mcpu=cortex-m4 -mthumb -nostartfiles -Wl,--gc-sections -Wl,-e,_start \
  -Wl,-Ttext=0x08000000 -Wl,-Tbss=0x20000000 \
  --specs=nosys.specs --specs=nano.specs -Wl,-Map=image.map
echo "    linked"

echo "==> Image size"
arm-none-eabi-size image.elf

echo "==> Allocator audit"
# The symbol table is dumped to a file rather than piped: `nm | grep -q` under
# `set -o pipefail` fails whenever grep matches early enough to exit before nm
# finishes writing, because nm then dies of SIGPIPE and supplies the
# pipeline's status. An audit that fails at random on a correct image is worse
# than no audit.
arm-none-eabi-nm image.elf > symbols.txt
if grep -iqE " (T|t|W) .*(malloc|calloc|realloc|_sbrk|newObj|nimGC)" symbols.txt; then
  echo "FAIL: an allocator is reachable from the linked image"
  exit 1
fi
echo "    no allocator linked in"

echo "==> Placement audit"
# Weights must be in .rodata (flash), the arena in .bss (RAM). Getting this
# backwards is silent and expensive: it works, and it costs you all your RAM.
if ! grep -qE "^0800.* R steady_.*_w_" symbols.txt; then
  echo "FAIL: weights are not in .rodata"
  exit 1
fi
echo "    weights in flash (.rodata)"
# Activation tables are constants too, and are just as expensive to get wrong:
# a 256-entry table copied into RAM at startup is 256 bytes of SRAM for nothing.
if ! grep -qE "^0800.* R steady_.*_lut" symbols.txt; then
  echo "FAIL: activation tables are not in .rodata"
  exit 1
fi
if ! grep -qE "^0800.* R steady_.*_exp" symbols.txt; then
  echo "FAIL: the softmax exp table is not in .rodata"
  exit 1
fi
echo "    activation and exp tables in flash (.rodata)"
if ! grep -qE "^2000.* B arena" symbols.txt; then
  echo "FAIL: arena is not in .bss"
  exit 1
fi
echo "    arena in RAM (.bss)"

echo
echo "freestanding check passed"
