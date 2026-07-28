# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# B-L475E-IOT01A: STM32L475VG, Cortex-M4 at 80 MHz, 96 KB of usable SRAM.
# Sourced by check.sh; see the contract at the top of that file.

BOARD_LABEL="STM32L475VG @ 80 MHz (B-L475E-IOT01A)"
BOARD_MCPU="-mcpu=cortex-m4 -mthumb -mfloat-abi=soft"
BOARD_SOURCES="board.c"
BOARD_LD="board.ld"
BOARD_RODATA_RE="^0800"

# Flashing and reset go through pyocd, which knows this board by name. The
# ST-LINK's drag-and-drop mass storage also works and needs no tools at all,
# but it cannot reset on demand, cannot read a register back, and cannot tell
# you that your stack pointer is in unmapped memory — which is how the first
# bring-up of this port was spent.
#
# Resolved here rather than inside board_probe, and that is not a style
# preference: check.sh runs the probe inside a command substitution to capture
# its reason for skipping, which is a subshell, so anything the probe assigns
# is discarded the moment it returns. Falling back to the project's virtualenv
# from in there left `pyocd` resolved during the probe and not found during the
# flash, which is a confusing way to fail four steps later.
PYOCD="${STEADY_MCU_PYOCD:-pyocd}"
command -v "$PYOCD" >/dev/null 2>&1 || PYOCD="$ROOT/.venv/bin/pyocd"

board_probe() {
  if ! command -v "$PYOCD" >/dev/null 2>&1; then
    echo "    pyocd not found; pip install pyocd, and give the probe a udev rule:"
    echo '    SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="374b", MODE="0666", TAG+="uaccess"'
    return 1
  fi
  "$PYOCD" list 2>/dev/null | grep -q stm32l475 || {
    echo "    no B-L475E-IOT01A found by pyocd"
    return 1
  }
  [ -e "$(board_console)" ] || {
    echo "    $(board_console) not present; set STEADY_MCU_TTY if the virtual COM port is elsewhere"
    return 1
  }
  return 0
}

board_console() { echo "${STEADY_MCU_TTY:-/dev/ttyACM0}"; }

board_flash() {
  local out="$1"
  echo "    flashing $(stat -c%s "$out/image.bin") bytes"
  "$PYOCD" flash -t stm32l475xg "$out/image.elf" 2>&1 | tail -1 | sed 's/^/    /'
}
