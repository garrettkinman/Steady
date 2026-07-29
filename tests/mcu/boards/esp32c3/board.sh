# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# ESP32-C3-DevKitM-1: ESP32-C3, a single RISC-V core at 160 MHz, 384 KB of
# usable SRAM, 4 MB of flash on the module.
# Sourced by check.sh; see the contract at the top of that file.
#
# The third board, and the first that is not a Cortex-M. Everything above this
# directory was already parameterised over the part; what this port added to
# the harness was a cross compiler prefix, a Nim --cpu, and a hook for turning
# a linked image into whatever the flasher wants — because on this part that is
# not one file.
#
# The image is in three pieces, and the split is forced by the hardware rather
# than chosen. The core cannot address flash: it addresses two windows that a
# page table maps onto it, and nothing maps them at reset. So the pieces are
#
#   0x00_0000  app.bin     the boot ROM loads this into SRAM and jumps to it —
#                          startup, board support, and nothing else
#   0x10_0000  rodata.bin  weights, read at 0x3C10_0000 once mapped
#   0x20_0000  text.bin    kernels, fetched at 0x4220_0000 once mapped
#
# board.c maps flash to those addresses one-to-one, so the offsets here and the
# addresses in board.ld are the same numbers; see both.

BOARD_LABEL="ESP32-C3 @ 160 MHz (ESP32-C3-DevKitM-1)"
BOARD_NIMCPU="riscv32"
BOARD_MCPU="-march=rv32imc -mabi=ilp32"
BOARD_SOURCES="board.c"
BOARD_STARTUP="startup.c"
BOARD_LD="board.ld"
# Weights live in the data window at 0x3C10_0000. SRAM on this part is 0x3FC9…,
# which shares a first character with nothing else and differs by the second —
# so the audit tests three, not two, and a weight array that quietly landed in
# RAM is still caught.
BOARD_RODATA_RE="^3c1"

# Espressif's RISC-V toolchain is not on PATH after an esp-idf install; it is
# under ~/.espressif with the version in the path. Resolved here rather than
# inside board_probe for the reason the STM32 port's pyocd is: check.sh runs
# the probe in a command substitution, so anything the probe assigns is
# discarded, and a compiler found during the probe and missing during the build
# is a confusing way to fail four steps later.
BOARD_CROSS="riscv32-esp-elf-"
if ! command -v "${BOARD_CROSS}gcc" >/dev/null 2>&1; then
  for d in "$HOME"/.espressif/tools/riscv32-esp-elf/*/riscv32-esp-elf/bin; do
    [ -x "$d/riscv32-esp-elf-gcc" ] && { BOARD_CROSS="$d/riscv32-esp-elf-"; break; }
  done
fi

# Nothing to add to the link. The two ARM boards ask for newlib's nosys and
# nano specs; this toolchain's defaults already produce a freestanding image,
# and asking for nano.specs here would fail rather than be ignored.
BOARD_LDFLAGS=""

ESPTOOL="${STEADY_MCU_ESPTOOL:-esptool}"
command -v "$ESPTOOL" >/dev/null 2>&1 || ESPTOOL="$ROOT/.venv/bin/esptool"

# ~~ the contract ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

board_console() { echo "${STEADY_MCU_TTY:-/dev/ttyUSB0}"; }

board_probe() {
  if ! command -v "${BOARD_CROSS}gcc" >/dev/null 2>&1; then
    echo "    riscv32-esp-elf-gcc not found; install esp-idf's tools, or set PATH"
    return 1
  fi
  if ! command -v "$ESPTOOL" >/dev/null 2>&1; then
    echo "    esptool not found; pip install esptool"
    return 1
  fi
  [ -e "$(board_console)" ] || {
    echo "    $(board_console) not present; set STEADY_MCU_TTY if the USB bridge is elsewhere"
    return 1
  }
  # The port existing is not the same as this part being behind it: the
  # DevKitM-1's CP2102N is a bridge chip that enumerates whatever it is
  # soldered to. Ask the ROM loader what it is, which also proves the reset
  # lines are wired the way the flasher will assume.
  #
  # Into a variable rather than through `| grep -q`, for the reason check.sh
  # gives at greater length about `nm`: grep exits on its first match, esptool
  # takes a SIGPIPE writing the rest, and under `set -o pipefail` the pipeline
  # then reports a failure that means only that the match was found early.
  local id
  id="$("$ESPTOOL" --port "$(board_console)" chip-id 2>/dev/null || true)"
  case "$id" in
    *ESP32-C3*) return 0 ;;
    *) echo "    nothing identifying as an ESP32-C3 answered on $(board_console)"
       return 1 ;;
  esac
}

# Three pieces rather than one, and the two mapped ones are cut out of the ELF
# before it is turned into a boot image: left in, they would be laid out by
# their addresses, and the gap between 0x3C10_0000 and 0x4220_0000 is 100 MB of
# padding.
board_image() {
  local out="$1"
  "${BOARD_CROSS}objcopy" -O binary -j .flash.rodata "$out/image.elf" "$out/rodata.bin"
  "${BOARD_CROSS}objcopy" -O binary -j .flash.text   "$out/image.elf" "$out/text.bin"
  "${BOARD_CROSS}objcopy" -R .flash.rodata -R .flash.text \
                          "$out/image.elf" "$out/ram.elf" 2>/dev/null

  # The flash mode and frequency are read out of this header by the boot ROM
  # and applied to the controller the cache reads through, so they configure
  # the memory system this benchmark is measuring — they are not packaging.
  "$ESPTOOL" --chip esp32c3 elf2image --flash-mode dio --flash-freq 80m \
             --flash-size 4MB -o "$out/image.bin" "$out/ram.elf" >/dev/null

  # And a linked address that disagrees with where board_flash writes the
  # bytes would be read as flash at the wrong offset: plausible rubbish rather
  # than a failure. The two agree here or the build stops.
  local a
  for a in "rodata 3c100000 .flash.rodata" "text 42200000 .flash.text"; do
    set -- $a
    "${BOARD_CROSS}readelf" -S "$out/image.elf" |
      grep -q "$3 *PROGBITS *$2" || {
        echo "FAIL: $3 is not linked at 0x$2, which is where board_flash writes it"
        return 1
      }
  done
}

board_flash() {
  local out="$1"
  echo "    flashing $(stat -c%s "$out/image.bin") + $(stat -c%s "$out/rodata.bin")" \
       "+ $(stat -c%s "$out/text.bin") bytes"
  "$ESPTOOL" --chip esp32c3 --port "$(board_console)" --baud 460800 \
             --after hard-reset write-flash \
             0x0 "$out/image.bin" \
             0x100000 "$out/rodata.bin" \
             0x200000 "$out/text.bin" 2>&1 |
    grep -E "^Wrote|^Hash|Hard resetting" | sed 's/^/    /'
}
