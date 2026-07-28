/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * STM32L475VG (B-L475E-IOT01A): 1 MB flash, 128 KB SRAM — but *not* 128 KB of
 * contiguous SRAM, which is the only thing here worth a comment.
 *
 * SRAM1 is 96 KB at 0x20000000. SRAM2 is a separate 32 KB at 0x10000000, and
 * on this part it is not aliased into the space above SRAM1: reads of
 * 0x20018000 and up fail on the bus. A single 128 KB region starting at
 * 0x20000000 therefore puts the initial stack pointer in nothing at all, and
 * the first push of the first function call takes a bus fault before a single
 * character can reach a UART. That was checked against the silicon with a
 * debugger rather than against a datasheet, after it happened.
 *
 * So: 96 KB, and every model this harness runs has to fit an arena plus a
 * stack in it. `fomo` is the largest at 78 KB of arena and does.
 */

MEMORY
{
  FLASH (rx)   : ORIGIN = 0x08000000, LENGTH = 1024K
  RAM   (rwx)  : ORIGIN = 0x20000000, LENGTH = 96K
  SRAM2 (rwx)  : ORIGIN = 0x10000000, LENGTH = 32K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

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
