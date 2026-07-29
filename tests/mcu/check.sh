#!/usr/bin/env bash
# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# On-target benchmark: build a firmware image per model, flash a board over
# USB, and read per-operator cycle counts back over its console.
#
# This is the only benchmark in the project, deliberately. Timing these models
# on a desktop core measures an out-of-order superscalar with megabytes of
# cache and other processes on it; the target is a Cortex-M4 reading its
# weights from flash through a few kilobytes of cache. Which kernel is faster
# is allowed to differ between them, and has — so the proxy was removed rather
# than kept alongside. See docs/performance.md.
#
#   tests/mcu/check.sh                        every model that fits
#   tests/mcu/check.sh kws vww                just these
#   tests/mcu/check.sh --board samd51 kws     on a named board
#
# Needs a board and the cross compiler that board asks for. Skips itself,
# loudly, without either.
#
# ~~ adding a board ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Everything part-specific lives under boards/<name>/, and everything above is
# shared. A port is three files.
#
# board.c, implementing what bench.nim.in calls and startup.c needs:
#
#   board_init        clock, console, counter; called before anything else
#   board_putc        one character to the console
#   board_poll        service the console; called between measured regions
#   board_led         a liveness signal, or nothing
#   board_cycles      a 32-bit counter ticking once per core cycle
#   board_sysclk      how many of those there are in a second
#   board_timebase    what that counter is, for the record
#   board_cache_configs / _label / _select / _state
#                     the sweep in front of flash, in this part's own terms
#
# a linker script, and a board.sh defining
#
#   BOARD_LABEL       what to print
#   BOARD_CROSS       cross-toolchain prefix, e.g. arm-none-eabi-
#   BOARD_NIMCPU      what to pass Nim's --cpu
#   BOARD_MCPU        architecture flags for the cross compiler
#   BOARD_LDFLAGS     extra flags for the link, if the toolchain wants any
#   BOARD_SOURCES     C files in the board directory, space separated
#   BOARD_STARTUP     startup file, if the shared Cortex-M one does not fit
#   BOARD_LD          linker script in the board directory
#   BOARD_RODATA_RE   what a flash address looks like in `nm` output, so the
#                     placement audit can tell flash from RAM on this part
#   board_probe       0 if the board is usable, else a printed reason
#   board_console     the path to read records from
#   board_image       turn $1/image.elf into whatever board_flash wants;
#                     the default is a flat binary and two of the three boards
#                     take it
#   board_flash       program it onto the board
#
# The three that exist differ about as much as three parts can. One is
# programmed through a debug probe and talks over a bridge chip's UART; one is
# programmed by copying a file onto a bootloader's mass-storage volume and
# talks over a CDC endpoint its own firmware implements; one is a different
# instruction set entirely, whose flash the core cannot address without first
# building a page table for it. None of that reaches any file above this line.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/tests/mcu"
BUILD="$ROOT/build/mcu"

BOARD="${STEADY_MCU_BOARD:-}"
MODELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --board) BOARD="$2"; shift 2 ;;
    --board=*) BOARD="${1#*=}"; shift ;;
    -h|--help) sed -n '7,21p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) MODELS+=("$1"); shift ;;
  esac
done

# Only models whose arena fits the smaller board's SRAM and whose weights fit
# its flash. mobilenet_v2 needs a 1.5 MB arena and is not a candidate on any
# part this compiler targets.
DEFAULT_MODELS="kws resnet8 vww person_detect fomo ad"
if [ ${#MODELS[@]} -eq 0 ]; then
  read -r -a MODELS <<<"${STEADY_MCU_MODELS:-$DEFAULT_MODELS}"
fi

# ~~ pick a board ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Defaults every board inherits and any board may override, defined before the
# first board.sh is sourced. The cross compiler is a board's own business now
# rather than this file's: with two instruction sets among three boards there
# is no single toolchain whose absence means "skip the benchmark", so each
# board reports its own missing compiler through board_probe like any other
# reason it cannot be used.

BOARD_CROSS="arm-none-eabi-"
BOARD_NIMCPU="arm"
BOARD_LDFLAGS="--specs=nosys.specs --specs=nano.specs"
BOARD_STARTUP=""                  # empty: the shared Cortex-M startup.c

board_image() { "${BOARD_CROSS}objcopy" -O binary "$1/image.elf" "$1/image.bin"; }

BOARDS=()
for d in "$DIR"/boards/*/; do BOARDS+=("$(basename "$d")"); done

if [ -n "$BOARD" ]; then
  [ -f "$DIR/boards/$BOARD/board.sh" ] || {
    echo "==> no such board '$BOARD'; have: ${BOARDS[*]}"
    exit 1
  }
  # shellcheck disable=SC1090
  source "$DIR/boards/$BOARD/board.sh"
  if ! reason="$(board_probe)"; then
    echo "==> $BOARD not usable; skipping the on-target benchmark"
    [ -n "$reason" ] && echo "$reason"
    exit 0
  fi
else
  # Nothing named, so find out. Refusing when two are attached is not
  # pedantry: numbers from two different parts are not comparable, and a table
  # that silently changed which one it came from is worse than no table.
  FOUND=(); REASONS=""
  for b in "${BOARDS[@]}"; do
    if ( source "$DIR/boards/$b/board.sh"; board_probe >/dev/null 2>&1 ); then
      FOUND+=("$b")
    else
      REASONS+="$(source "$DIR/boards/$b/board.sh"; board_probe 2>&1 || true)"$'\n'
    fi
  done
  if [ ${#FOUND[@]} -eq 0 ]; then
    echo "==> no supported board found; skipping the on-target benchmark"
    printf '%s' "$REASONS"
    exit 0
  fi
  if [ ${#FOUND[@]} -gt 1 ]; then
    echo "==> more than one supported board is attached: ${FOUND[*]}"
    echo "    name one with --board <name>; their numbers are not comparable"
    exit 1
  fi
  BOARD="${FOUND[0]}"
  # shellcheck disable=SC1090
  source "$DIR/boards/$BOARD/board.sh"
fi

BDIR="$DIR/boards/$BOARD"
NIMLIB="$(nim --hints:off --eval:'import std/os; echo getCurrentCompilerExe().parentDir.parentDir / "lib"' 2>/dev/null | tail -1)"

echo "==> Building the compiler"
nim c --hints:off -d:release --path:"$ROOT/src" \
      --out:"$BUILD/steadyc" "$ROOT/src/steadyc.nim"
echo "==> Target  $BOARD — $BOARD_LABEL"

CFLAGS="-c -Os -ffreestanding -ffunction-sections -fdata-sections -Wall $BOARD_MCPU"

# The shared startup, unless the board brought one. What is shared is a
# Cortex-M vector table and reset handler, so a board on another instruction
# set supplies its own rather than that file growing a `#if` about which core
# it is being compiled for.
STARTUP="${BOARD_STARTUP:+$BDIR/$BOARD_STARTUP}"
STARTUP="${STARTUP:-$DIR/startup.c}"

for name in "${MODELS[@]}"; do
  model="$ROOT/tests/models/$name.tflite"
  [ -f "$model" ] || { echo "==> $name: not fetched, skipping"; continue; }

  echo
  echo "==> $name"
  out="$BUILD/$BOARD/$name"
  rm -rf "$out"; mkdir -p "$out"

  # Identifies this build in the record it produces; see bench.nim.in.
  NONCE="$(date +%s)$RANDOM"

  "$BUILD/steadyc" "$model" -o "$out" -n "$name" --no-capi -q
  sed "s/@MODEL@/$name/g" "$DIR/bench.nim.in" > "$out/bench.nim"

  # Nim to C only; the cross compiler does the rest, exactly as the
  # freestanding check does.
  nim c --hints:off --path:"$ROOT/src" --path:"$ROOT/tests/freestanding" \
        --os:any --cpu:"$BOARD_NIMCPU" --mm:none --panics:on --noMain \
        --compileOnly --nimcache:"$out/nimcache" \
        -d:danger -d:steadyProfile -d:useMalloc -d:steadyNonce="$NONCE" \
        "$out/bench.nim" >/dev/null

  CC="${BOARD_CROSS}gcc"
  ( cd "$out/nimcache"
    for f in *.c; do "$CC" $CFLAGS -w -I"$NIMLIB" -o "${f%.c}.o" "$f"; done )
  "$CC" $CFLAGS -w -I"$NIMLIB" -o "$out/weights.o" "$out/${name}_weights.c"
  "$CC" $CFLAGS -o "$out/startup.o" "$STARTUP"
  BOARD_OBJS=()
  for src in $BOARD_SOURCES; do
    "$CC" $CFLAGS -o "$out/${src%.c}.o" "$BDIR/$src"
    BOARD_OBJS+=("$out/${src%.c}.o")
  done

  "$CC" -o "$out/image.elf" \
    "$out/startup.o" "${BOARD_OBJS[@]}" "$out/weights.o" "$out"/nimcache/*.o \
    $BOARD_MCPU -nostartfiles -T"$BDIR/$BOARD_LD" \
    -Wl,--gc-sections -Wl,-Map="$out/image.map" \
    $BOARD_LDFLAGS

  board_image "$out"
  "${BOARD_CROSS}size" "$out/image.elf" | sed 's/^/    /'

  # Same audit the freestanding check runs. A benchmark image that quietly
  # linked an allocator would still produce numbers, and they would be numbers
  # for a different program than the one being shipped.
  #
  # The symbol table goes to a file first rather than into a pipe. Under
  # `set -o pipefail`, `nm | grep -q` reports failure whenever grep finds its
  # match early enough to exit before nm has finished writing: nm takes a
  # SIGPIPE, and its status becomes the pipeline's. The audit then fails at
  # random, on a correct image, which is worse than not auditing at all.
  "${BOARD_CROSS}nm" "$out/image.elf" > "$out/symbols.txt"
  if grep -iqE " (T|t|W) .*(malloc|calloc|realloc|_sbrk|newObj|nimGC)" "$out/symbols.txt"; then
    echo "FAIL: an allocator is reachable from the benchmark image"
    exit 1
  fi
  if ! grep -qE "$BOARD_RODATA_RE.* R steady_.*_w_" "$out/symbols.txt"; then
    echo "FAIL: weights are not in flash"
    exit 1
  fi
  echo "    no allocator linked in; weights in flash"

  board_flash "$out"

  if ! TTY="$(board_console)"; then
    echo "FAIL: no console appeared after flashing (see $out/)"
    exit 1
  fi
  echo "    console $TTY"

  # The firmware loops forever, so listening can start after programming
  # rather than racing it.
  #
  # The line discipline is worth setting and not worth waiting forever for.
  # On a board whose console is a USB CDC endpoint, `stty` is not a local
  # operation: it sends a control request and blocks until the device answers,
  # and this device answers between measurements — which on the largest model
  # is seconds away. Unbounded, it wedged a whole run behind one measurement of
  # `fomo`. The firmware now polls between repetitions, and this is the belt to
  # that pair of braces.
  timeout 15 stty -F "$TTY" 115200 raw -echo -hupcl 2>/dev/null || true
  # Drain whatever the port buffered from the previous image before listening.
  timeout 2 cat "$TTY" > /dev/null 2>&1 || true
  : > "$out/serial.txt"

  # Wait for two complete records and report the last. The firmware repeats
  # forever, so this costs a few seconds and removes every first-run effect
  # at once: a capture that started mid-record, and caches that are cold on
  # the first pass in a way they never are again.
  #
  # The reader is restarted if it stops, and appends rather than truncates. A
  # console that is itself a USB device can disappear and come back — that is
  # what reflashing one looks like from here — and a reader that quietly died
  # early would otherwise present as a timeout with an empty capture and
  # nothing to say about why.
  deadline=$((SECONDS + 300))
  CATPID=
  trap 'kill $CATPID 2>/dev/null || true' EXIT
  while [ $SECONDS -lt $deadline ]; do
    if [ -z "$CATPID" ] || ! kill -0 "$CATPID" 2>/dev/null; then
      timeout 300 cat "$TTY" >> "$out/serial.txt" 2>/dev/null &
      CATPID=$!
    fi
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
echo "on-target benchmark complete; raw captures under build/mcu/$BOARD/<model>/"
