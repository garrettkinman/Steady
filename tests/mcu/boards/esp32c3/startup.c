/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * Minimal RISC-V startup, in place of the Cortex-M one the other two boards
 * share: a stack, a global pointer, a zeroed .bss, and a trap handler that
 * parks. See board.ld for what the boot ROM has already done by the time this
 * runs, and what it deliberately has not.
 *
 * No interrupt is enabled anywhere in this firmware — mie is cleared and
 * mstatus.MIE with it — for the reason the Cortex-M startup gives: a timer
 * tick landing inside a measured region is exactly the noise a cycle counter
 * is supposed to be free of. On this part that is not merely a default worth
 * keeping. The boot ROM runs with interrupts on, so it has to be undone.
 */

#include <stdint.h>

extern uint32_t _sbss, _ebss;

void board_init(void);
void steady_mcu_main(void);

void reset_handler(void);

/* The image entry point. `sp` and `gp` have to be established before any
   compiler-generated code runs, which is what makes this naked and what makes
   it the one piece of assembly in the port. `norelax` around the `gp` load is
   not decoration: with relaxation on, the assembler rewrites `la gp, ...` into
   an access relative to the `gp` being loaded, which loads gp from itself. */
__attribute__((naked, section(".init.entry"), used))
void _start(void) {
  __asm__ volatile(
      ".option push\n"
      ".option norelax\n"
      "  la gp, __global_pointer$\n"
      ".option pop\n"
      "  la sp, _estack\n"
      "  j  reset_handler\n");
}

/* Every trap parks here rather than returning. Nothing in this image should
   take one, so arriving is a bug worth being able to see in a debugger — and
   an infinite loop at a known address is easier to recognise than a jump to
   whatever the boot ROM last left in mtvec.
 *
 * The alignment is load-bearing and 4 is not enough. This core ignores the
 * low eight bits of whatever is written to mtvec and forces the mode field to
 * vectored: a handler at 0x4038_02C0 comes back out of the register as
 * 0x4038_0201, so traps land 0xC0 short of it, in the middle of some other
 * function. That was not theoretical — it is what an earlier version of this
 * file did, and the symptom was that a deliberately illegal instruction
 * restarted the firmware instead of parking it. Aligning to the 256 bytes the
 * register actually keeps makes the base survive the write, and vectored mode
 * sends exceptions to the base, which is here. */
__attribute__((naked, aligned(256)))
static void trap_handler(void) {
  __asm__ volatile("1: j 1b");
}

void reset_handler(void) {
  /* Global disable, and only that. The obvious `csrw mie, 0` alongside it
     traps as an illegal instruction on this core: it routes its interrupts
     through an external controller and does not implement the standard `mie`
     register at all, which is also why esp-idf's own RISC-V support touches
     mstatus and never mie. Clearing MIE is sufficient regardless — no
     interrupt can be taken in machine mode with it clear. */
  __asm__ volatile("csrci mstatus, 8");   /* MIE */
  __asm__ volatile("csrw mtvec, %0" :: "r"(&trap_handler));

  /* .data needs no copying here. The other two boards' startup moves it out of
     flash because their images are executed in place; this one is loaded into
     RAM segment by segment by the boot ROM, which has already put .data where
     it belongs. .bss is not in the image at all, so it is still ours to clear. */
  for (uint32_t *p = &_sbss; p < &_ebss;) *p++ = 0;

  board_init();
  steady_mcu_main();
  for (;;) { }
}
