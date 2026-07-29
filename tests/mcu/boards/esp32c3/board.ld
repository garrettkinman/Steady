/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * ESP32-C3-DevKitM-1 memory map.
 *
 * This part reaches flash only through a cache, and the cache is not mapped
 * when the image starts, so the layout is in two halves rather than one.
 *
 *   loaded    the boot ROM copies these segments out of flash into SRAM and
 *             jumps to the entry point. Startup and board support live here,
 *             because they are what programs the MMU, and code cannot fetch
 *             itself through a mapping that does not exist yet.
 *   mapped    everything else. Kernels and weights are read from flash
 *             through the cache at addresses this script fixes and board.c
 *             maps one-to-one — vaddr 0x3C00_0000 + n is flash byte n. Neither
 *             is in the image the ROM loads; board.sh writes them to the flash
 *             offsets named here.
 *
 * That split is the whole reason this board has a startup.c of its own. The
 * two Cortex-M ports execute in place out of a flash the core can simply
 * address, and have nothing to arrange before main.
 *
 * SRAM is one 384 KB block with two windows onto it, and they are the same
 * bytes: instruction address 0x4038_0000 + n and data address 0x3FC8_0000 + n
 * are one location. The two regions below are therefore carved out of one
 * range, not two, and must not overlap when expressed in either window. The
 * top 64 KB is left to the boot ROM, whose stack and data are still live.
 */

ENTRY(_start)

MEMORY
{
  /* 0x4038_0000..0x4039_0000, i.e. data 0x3FC8_0000..0x3FC9_0000 */
  iram (rwx) : ORIGIN = 0x40380000, LENGTH = 64K
  /* 0x3FC9_0000..0x3FCD_0000, i.e. instruction 0x4039_0000..0x403D_0000 */
  dram (rw)  : ORIGIN = 0x3FC90000, LENGTH = 256K
  /* Flash, through the cache. The offsets 0x10_0000 and 0x20_0000 are where
     board.sh writes these two, and the addresses below say so: identity
     mapping means the low 23 bits of a mapped address are its flash offset. */
  drom (r)   : ORIGIN = 0x3C100000, LENGTH = 1M
  irom (rx)  : ORIGIN = 0x42200000, LENGTH = 1M
}

SECTIONS
{
  /* ~~ loaded by the boot ROM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  .iram.text : ALIGN(4)
  {
    KEEP(*(.init.entry))
    /* Everything that runs before, or while, the mapping is being set up.
       Named by object file rather than by section attribute because that is
       the actual requirement — it is not a property of a function that it
       must not be fetched through the cache it is programming. */
    *startup.o(.text .text.*)
    *board.o(.text .text.*)
    /* And anything that asks, by section attribute, for the same treatment. */
    *(.iram.text .iram.text.*)
    . = ALIGN(4);
  } > iram

  .iram.rodata : ALIGN(4)
  {
    /* Board strings and jump tables, for the same reason: board_init reads
       them, and it runs before flash is readable. */
    *startup.o(.rodata .rodata.* .srodata .srodata.*)
    *board.o(.rodata .rodata.* .srodata .srodata.*)
    . = ALIGN(4);
  } > iram

  .data : ALIGN(4)
  {
    /* No load-address dance and no copy loop in startup: the ROM loads this
       segment straight to where it is linked. */
    *(.data .data.*)
    . = ALIGN(4);
    PROVIDE(__global_pointer$ = . + 0x800);
    *(.sdata .sdata.* .sdata2 .sdata2.*)
    . = ALIGN(4);
  } > dram

  .bss (NOLOAD) : ALIGN(4)
  {
    _sbss = .;
    *(.sbss .sbss.* .scommon)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > dram

  /* ~~ mapped from flash ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  .flash.rodata : ALIGN(4)
  {
    _flash_rodata_start = .;
    /* The weights, the activation tables, and the benchmark's own literals.
       The placement audit in check.sh reads the addresses back out of the
       symbol table and fails the build if a weight array is not here. */
    *(.rodata .rodata.* .srodata .srodata.*)
    . = ALIGN(4);
    _flash_rodata_end = .;
  } > drom

  .flash.text : ALIGN(4)
  {
    _flash_text_start = .;
    *(.text .text.*)
    . = ALIGN(4);
    _flash_text_end = .;
  } > irom

  _estack = ORIGIN(dram) + LENGTH(dram);

  /DISCARD/ : { *(.eh_frame) *(.comment) *(.riscv.attributes) }
}
