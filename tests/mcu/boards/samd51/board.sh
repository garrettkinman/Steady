# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Adafruit ItsyBitsy M4 Express: ATSAMD51G19A, Cortex-M4 at 120 MHz, 192 KB of
# SRAM, 512 KB of flash less the 16 KB the bootloader keeps.
# Sourced by check.sh; see the contract at the top of that file.
#
# Nothing about this board is reached through a debug probe, because it does
# not have one. Both directions go over the one USB cable:
#
#   in   the resident UF2 bootloader, which appears as a mass-storage volume
#        and programs whatever is copied onto it
#   out  a CDC serial port the benchmark firmware serves itself
#
# and the hinge between them is the 1200-baud open-and-close every
# Arduino-compatible bootloader treats as "reboot into the bootloader". Both
# the firmware here and the CircuitPython that ships on the board honour it, so
# the harness can reflash between models without anyone pressing reset.

BOARD_LABEL="ATSAMD51G19A @ 120 MHz (Adafruit ItsyBitsy M4 Express)"
BOARD_MCPU="-mcpu=cortex-m4 -mthumb -mfloat-abi=soft"
BOARD_SOURCES="board.c usb_cdc.c"
BOARD_LD="board.ld"
# The image is linked at 0x4000, above the bootloader, so weights land in the
# low 512 KB with a leading zero rather than at the STM32's 0x0800_0000.
BOARD_RODATA_RE="^0000"

SAMD51_APP_ID="1209:0001"        # the benchmark firmware's own USB identity
SAMD51_UF2_FAMILY="0x55114460"   # SAMD51, as the bootloader checks it
SAMD51_APP_BASE="0x4000"

# ~~ finding things ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

samd51_tty_id() {   # /dev/ttyACMn -> "vid:pid"
  udevadm info -q property -n "$1" 2>/dev/null | awk -F= '
    $1 == "ID_VENDOR_ID" { v = $2 } $1 == "ID_MODEL_ID" { p = $2 }
    END { if (v != "" && p != "") print v ":" p }'
}

samd51_app_tty() {
  local t
  for t in /dev/ttyACM*; do
    [ -e "$t" ] || continue
    [ "$(samd51_tty_id "$t")" = "$SAMD51_APP_ID" ] && { echo "$t"; return 0; }
  done
  return 1
}

# Any port this board might currently be presenting: ours once it has been
# flashed, or whatever Adafruit firmware is still on it the first time.
samd51_any_tty() {
  local t
  samd51_app_tty && return 0
  for t in /dev/ttyACM*; do
    [ -e "$t" ] || continue
    case "$(samd51_tty_id "$t")" in 239a:*) echo "$t"; return 0 ;; esac
  done
  return 1
}

# A mounted UF2 bootloader volume identifies itself by the file it puts there,
# which is more reliable than its label — the label is per-board, INFO_UF2.TXT
# is per-format.
samd51_mounted_uf2() {
  local m
  for m in /media/*/* /run/media/*/* /media/* /mnt/*; do
    [ -f "$m/INFO_UF2.TXT" ] && { echo "$m"; return 0; }
  done
  return 1
}

samd51_uf2_drive() {
  local line NAME LABEL MOUNTPOINT mp
  samd51_mounted_uf2 && return 0
  # Present but not mounted: ask udisks, which needs no privileges for
  # removable media and puts it where a desktop session would have.
  while read -r line; do
    NAME=; LABEL=; MOUNTPOINT=
    eval "$line"
    case "$LABEL" in *BOOT) ;; *) continue ;; esac
    if [ -n "$MOUNTPOINT" ]; then echo "$MOUNTPOINT"; return 0; fi
    mp="$(udisksctl mount -b "/dev/$NAME" 2>/dev/null |
            sed -n 's/^Mounted .* at \(.*\)$/\1/p' | sed 's/\.$//')"
    [ -n "$mp" ] && { echo "$mp"; return 0; }
  done < <(lsblk -Pno NAME,LABEL,MOUNTPOINT 2>/dev/null)
  return 1
}

# Open the port at 1200 baud and close it, which drops DTR. Holding a second
# descriptor across the stty is what keeps stty's own close from being the
# last one — the hangup has to happen after the baud rate has been set, not
# during it.
samd51_touch_1200() {
  local tty="$1"
  exec 9<>"$tty" 2>/dev/null || return 1
  stty -F "$tty" 1200 hupcl >/dev/null 2>&1 || true
  exec 9>&-
  return 0
}

# ~~ the contract ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

board_probe() {
  command -v "${BOARD_CROSS}gcc" >/dev/null 2>&1 || {
    echo "    ${BOARD_CROSS}gcc not found; this board needs the ARM cross compiler"
    return 1
  }
  command -v udevadm >/dev/null 2>&1 || {
    echo "    udevadm not found; needed to tell this board's serial port from any other"
    return 1
  }
  if samd51_mounted_uf2 >/dev/null || samd51_any_tty >/dev/null; then
    return 0
  fi
  echo "    no ItsyBitsy M4 found: no UF2 bootloader volume and no serial port from one"
  return 1
}

# After flashing, the board re-enumerates as the benchmark firmware. Nothing
# else on the machine claims 1209:0001, so waiting for that identity to appear
# is both the way to find the port and the way to know the image booted.
board_console() {
  local t deadline=$((SECONDS + 30))
  while [ $SECONDS -lt $deadline ]; do
    t="$(samd51_app_tty || true)"
    [ -n "$t" ] && { echo "$t"; return 0; }
    sleep 0.5
  done
  echo "/dev/null"
  return 1
}

board_flash() {
  local out="$1" drive tty i

  python3 "$DIR/uf2conv.py" --base "$SAMD51_APP_BASE" \
          --family "$SAMD51_UF2_FAMILY" -o "$out/image.uf2" "$out/image.bin"

  drive="$(samd51_uf2_drive || true)"
  if [ -z "$drive" ]; then
    tty="$(samd51_any_tty || true)"
    if [ -n "$tty" ]; then
      echo "    asking $tty for the bootloader (1200 baud, DTR dropped)"
      samd51_touch_1200 "$tty"
    else
      echo "    no serial port to ask through; double-tap RESET on the board"
    fi
    for i in $(seq 1 60); do
      sleep 0.5
      drive="$(samd51_uf2_drive || true)"
      [ -n "$drive" ] && break
    done
  fi
  [ -n "$drive" ] || {
    echo "FAIL: no UF2 bootloader volume appeared; double-tap RESET and rerun"
    return 1
  }

  echo "    copying $(stat -c%s "$out/image.uf2") bytes to $drive"
  # The copy is expected to end badly: the bootloader programs as the blocks
  # arrive and resets the moment it has the last one, so the volume vanishes
  # underneath the write. A failed close here is the success case, and the
  # thing that actually decides whether it worked is whether a port carrying
  # this build's nonce shows up afterwards.
  cp "$out/image.uf2" "$drive/" 2>/dev/null || true
  sync 2>/dev/null || true
  sleep 1
}
