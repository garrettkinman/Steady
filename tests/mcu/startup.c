/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * Minimal Cortex-M startup: a vector table, a reset handler that puts .data
 * and .bss where the C runtime expects them, and nothing else. No interrupts
 * are enabled anywhere in this firmware — a timer tick landing in the middle
 * of a measured region is exactly the noise a cycle counter is supposed to be
 * free of.
 */

#include <stdint.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;

void board_init(void);
void steady_mcu_main(void);

void Reset_Handler(void);
static void Default_Handler(void) { for (;;) { } }

__attribute__((section(".isr_vector"), used))
void (*const g_vectors[])(void) = {
    (void (*)(void)) & _estack,
    Reset_Handler,
    Default_Handler,   /* NMI          */
    Default_Handler,   /* HardFault    */
    Default_Handler,   /* MemManage    */
    Default_Handler,   /* BusFault     */
    Default_Handler,   /* UsageFault   */
    0, 0, 0, 0,
    Default_Handler,   /* SVCall       */
    Default_Handler,   /* DebugMonitor */
    0,
    Default_Handler,   /* PendSV       */
    Default_Handler,   /* SysTick      */
};

void Reset_Handler(void) {
  uint32_t *src = &_sidata;
  uint32_t *dst = &_sdata;
  while (dst < &_edata) *dst++ = *src++;
  for (dst = &_sbss; dst < &_ebss;) *dst++ = 0;

  board_init();
  steady_mcu_main();
  for (;;) { }
}
