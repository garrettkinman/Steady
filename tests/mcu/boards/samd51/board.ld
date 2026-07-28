/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * ATSAMD51G19A (Adafruit ItsyBitsy M4 Express): 512 KB flash, 192 KB SRAM.
 *
 * Two things here are not the datasheet's numbers.
 *
 * The image starts at 0x4000, not 0. The first 16 KB of flash hold the UF2
 * bootloader, which is also the only way onto this board — it has no debug
 * probe, and overwriting it would leave nothing to flash the replacement
 * with. `startup.c` moves VTOR to match.
 *
 * The stack starts eight bytes below the top of SRAM rather than at it. The
 * bootloader's double-tap flag lives in the last word of RAM, and the
 * firmware writes it there deliberately to ask for the bootloader on the next
 * reset — which is how the harness reflashes without anyone pressing a
 * button. A stack based at the very top would have the first push of the
 * first call land on that word.
 */

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00004000, LENGTH = 496K
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 192K
}

_estack = ORIGIN(RAM) + LENGTH(RAM) - 8;

/* Where the bootloader looks for the double-tap request. */
_dbl_tap = ORIGIN(RAM) + LENGTH(RAM) - 4;

SECTIONS
{
  .isr_vector : {
    . = ALIGN(4);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } > FLASH

  .text : {
    . = ALIGN(4);
    *(.text*)
    . = ALIGN(4);
  } > FLASH

  /* Weights and activation tables are const arrays and belong here, in flash.
     This is a separate output section rather than part of .text so that the
     placement audit can still tell the two apart: `nm` types a symbol by the
     section holding it, and folding rodata into .text would report every
     weight array as code and quietly make the audit unfalsifiable. */
  .rodata : {
    . = ALIGN(4);
    *(.rodata*)
    . = ALIGN(4);
  } > FLASH

  _sidata = LOADADDR(.data);
  .data : {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > RAM AT> FLASH

  .bss : {
    . = ALIGN(8);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(8);
    _ebss = .;
  } > RAM

  /* The arena lives in .bss and is 8-byte aligned by the generated module. */

  /DISCARD/ : {
    *(.ARM.exidx*)
    *(.ARM.extab*)
  }
}
