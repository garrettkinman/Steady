#!/usr/bin/env bash
# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# On-target benchmark: build a firmware image per model, flash a
# B-L475E-IOT01A over its ST-LINK, and read per-operator cycle counts back
# over the virtual COM port.
#
# This is the measurement the host benchmark is a proxy for. The host runs an
# out-of-order superscalar core with megabytes of cache; the target is an
# 80 MHz Cortex-M4 reading its weights from flash through an 8-line data
# cache. Those are different machines, and which kernel is faster is allowed
# to differ between them — the whole point of this script is that it does not
# have to be guessed.
#
#   tests/mcu/check.sh              every model that fits
#   tests/mcu/check.sh kws vww      just these
#
# Needs arm-none-eabi-gcc and a board. Skips itself, loudly, without either.
#
# Flashing and reset go through pyocd, which knows this board by name. The
# ST-LINK's drag-and-drop mass storage also works and needs no tools at all,
# but it cannot reset on demand, cannot read a register back, and cannot tell
# you that your stack pointer is in unmapped memory — which is how the first
# bring-up of this file was spent. Results come back over /dev/ttyACM0 at
# 115200.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/tests/mcu"
BUILD="$ROOT/build/mcu"
TTY="${STEADY_MCU_TTY:-/dev/ttyACM0}"

# Only models whose arena fits 96 KB of SRAM1 and whose weights fit 1 MB of
# flash. mobilenet_v2 needs a 1.5 MB arena and is not a candidate on any part
# this compiler targets.
DEFAULT_MODELS="kws resnet8 vww person_detect fomo ad"
MODELS=("${@:-}")
if [ -z "${MODELS[0]:-}" ]; then
  read -r -a MODELS <<<"${STEADY_MCU_MODELS:-$DEFAULT_MODELS}"
fi

if ! command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  echo "==> arm-none-eabi-gcc not found; skipping the on-target benchmark"
  exit 0
fi

PYOCD="${STEADY_MCU_PYOCD:-pyocd}"
command -v "$PYOCD" >/dev/null 2>&1 || PYOCD="$ROOT/.venv/bin/pyocd"
if ! command -v "$PYOCD" >/dev/null 2>&1; then
  echo "==> pyocd not found; skipping the on-target benchmark"
  echo "    pip install pyocd, and give the probe a udev rule:"
  echo '    SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="374b", MODE="0666", TAG+="uaccess"'
  exit 0
fi
if ! "$PYOCD" list 2>/dev/null | grep -q stm32l475; then
  echo "==> no B-L475E-IOT01A found by pyocd; skipping the on-target benchmark"
  exit 0
fi
if [ ! -e "$TTY" ]; then
  echo "==> $TTY not present; skipping the on-target benchmark"
  echo "    set STEADY_MCU_TTY if the virtual COM port is elsewhere"
  exit 0
fi

NIMLIB="$(nim --hints:off --eval:'import std/os; echo getCurrentCompilerExe().parentDir.parentDir / "lib"' 2>/dev/null | tail -1)"

echo "==> Building the compiler"
nim c --hints:off -d:release --path:"$ROOT/src" \
      --out:"$BUILD/steadyc" "$ROOT/src/steadyc.nim"
echo "==> Target  STM32L475VG @ 80 MHz   console $TTY"

MCPU="-mcpu=cortex-m4 -mthumb -mfloat-abi=soft"
CFLAGS="-c -Os -ffreestanding -ffunction-sections -fdata-sections -Wall $MCPU"

for name in "${MODELS[@]}"; do
  model="$ROOT/tests/models/$name.tflite"
  [ -f "$model" ] || { echo "==> $name: not fetched, skipping"; continue; }

  echo
  echo "==> $name"
  out="$BUILD/$name"
  rm -rf "$out"; mkdir -p "$out"

  # Identifies this build in the record it produces; see bench.nim.in.
  NONCE="$(date +%s)$RANDOM"

  "$BUILD/steadyc" "$model" -o "$out" -n "$name" --no-capi -q
  sed "s/@MODEL@/$name/g" "$DIR/bench.nim.in" > "$out/bench.nim"

  # Nim to C only; the cross compiler does the rest, exactly as the
  # freestanding check does.
  nim c --hints:off --path:"$ROOT/src" --path:"$ROOT/tests/freestanding" \
        --os:any --cpu:arm --mm:none --panics:on --noMain \
        --compileOnly --nimcache:"$out/nimcache" \
        -d:danger -d:steadyProfile -d:useMalloc -d:steadyNonce="$NONCE" \
        "$out/bench.nim" >/dev/null

  ( cd "$out/nimcache"
    for f in *.c; do arm-none-eabi-gcc $CFLAGS -w -I"$NIMLIB" -o "${f%.c}.o" "$f"; done )
  arm-none-eabi-gcc $CFLAGS -w -I"$NIMLIB" -o "$out/weights.o" "$out/${name}_weights.c"
  arm-none-eabi-gcc $CFLAGS -o "$out/startup.o" "$DIR/startup.c"
  arm-none-eabi-gcc $CFLAGS -o "$out/board.o" "$DIR/board.c"

  arm-none-eabi-gcc -o "$out/image.elf" \
    "$out/startup.o" "$out/board.o" "$out/weights.o" "$out"/nimcache/*.o \
    $MCPU -nostartfiles -T"$DIR/stm32l475.ld" \
    -Wl,--gc-sections -Wl,-Map="$out/image.map" \
    --specs=nosys.specs --specs=nano.specs

  arm-none-eabi-objcopy -O binary "$out/image.elf" "$out/image.bin"
  arm-none-eabi-size "$out/image.elf" | sed 's/^/    /'

  # Same audit the freestanding check runs. A benchmark image that quietly
  # linked an allocator would still produce numbers, and they would be numbers
  # for a different program than the one being shipped.
  #
  # The symbol table goes to a file first rather than into a pipe. Under
  # `set -o pipefail`, `nm | grep -q` reports failure whenever grep finds its
  # match early enough to exit before nm has finished writing: nm takes a
  # SIGPIPE, and its status becomes the pipeline's. The audit then fails at
  # random, on a correct image, which is worse than not auditing at all.
  arm-none-eabi-nm "$out/image.elf" > "$out/symbols.txt"
  if grep -iqE " (T|t|W) .*(malloc|calloc|realloc|_sbrk|newObj|nimGC)" "$out/symbols.txt"; then
    echo "FAIL: an allocator is reachable from the benchmark image"
    exit 1
  fi
  if ! grep -qE "^0800.* R steady_.*_w_" "$out/symbols.txt"; then
    echo "FAIL: weights are not in flash"
    exit 1
  fi
  echo "    no allocator linked in; weights in flash"

  echo "    flashing $(stat -c%s "$out/image.bin") bytes"
  "$PYOCD" flash -t stm32l475xg "$out/image.elf" 2>&1 | tail -1 | sed 's/^/    /'

  # The firmware loops forever, so listening can start after programming
  # rather than racing it.
  stty -F "$TTY" 115200 raw -echo -hupcl 2>/dev/null || true
  # Drain whatever the port buffered from the previous image before listening.
  timeout 2 cat "$TTY" > /dev/null 2>&1 || true
  : > "$out/serial.txt"
  timeout 300 cat "$TTY" > "$out/serial.txt" 2>/dev/null &
  CATPID=$!
  trap 'kill $CATPID 2>/dev/null || true' EXIT

  # Wait for two complete records and report the last. The firmware repeats
  # forever, so this costs a few seconds and removes every first-run effect
  # at once: a capture that started mid-record, and caches that are cold on
  # the first pass in a way they never are again.
  deadline=$((SECONDS + 240))
  while [ $SECONDS -lt $deadline ]; do
    if [ "$(grep -ac '^END' "$out/serial.txt" 2>/dev/null || true)" -ge 2 ]; then
      break
    fi
    sleep 2
  done
  kill $CATPID 2>/dev/null || true
  wait $CATPID 2>/dev/null || true
  trap - EXIT

  if [ "$(grep -ac '^END' "$out/serial.txt" 2>/dev/null || true)" -lt 1 ]; then
    echo "FAIL: no complete record from the board (see $out/serial.txt)"
    exit 1
  fi

  python3 "$DIR/report.py" "$out/$name.nim" "$out/serial.txt" "$NONCE"
done

echo
echo "on-target benchmark complete; raw captures under build/mcu/<model>/"
