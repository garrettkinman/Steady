/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * B-L475E-IOT01A board support: clock, UART, cycle counter, and the flash
 * accelerator controls that the benchmark treats as an experimental variable.
 *
 * Registers are written by address rather than through a vendor HAL. This is
 * about a hundred lines against a dependency whose headers are larger than
 * this whole project, and every line of it is checkable against RM0351.
 */

#include <stdint.h>

#define REG(a) (*(volatile uint32_t *)(a))

/* RCC */
#define RCC_BASE     0x40021000u
#define RCC_CR       REG(RCC_BASE + 0x00)
#define RCC_CFGR     REG(RCC_BASE + 0x08)
#define RCC_PLLCFGR  REG(RCC_BASE + 0x0C)
#define RCC_AHB2ENR  REG(RCC_BASE + 0x4C)
#define RCC_APB1ENR1 REG(RCC_BASE + 0x58)
#define RCC_APB2ENR  REG(RCC_BASE + 0x60)

/* Flash interface — LATENCY[2:0], PRFTEN(8), ICEN(9), DCEN(10),
   ICRST(11), DCRST(12). The last four are the experiment. */
#define FLASH_ACR    REG(0x40022000u)

/* GPIO */
#define GPIOB_BASE   0x48000400u
#define GPIO_MODER(b) REG((b) + 0x00)
#define GPIO_OTYPER(b) REG((b) + 0x04)
#define GPIO_OSPEEDR(b) REG((b) + 0x08)
#define GPIO_BSRR(b)  REG((b) + 0x18)
#define GPIO_AFRL(b)  REG((b) + 0x20)

/* USART1 on PB6/PB7 is the ST-LINK virtual COM port on this board, confirmed
   against the hardware during bring-up rather than taken from a pinout table. */
#define USART1_BASE  0x40013800u
#define USART_CR1(b) REG((b) + 0x00)
#define USART_BRR(b) REG((b) + 0x0C)
#define USART_ISR(b) REG((b) + 0x1C)
#define USART_TDR(b) REG((b) + 0x28)

/* TIM2 is the time base, not the more obvious DWT cycle counter.
 *
 * `DWT->CYCCNT` is the usual answer and it is the wrong one here: the DWT
 * lives in the debug power domain, which is powered down whenever a probe is
 * not attached. Setting TRCENA and CYCCNTENA is not enough — the counter
 * simply stops when the debugger detaches, and a benchmark whose numbers are
 * valid only while being watched is worse than no benchmark. This was
 * diagnosed the entertaining way: every measurement read zero except the
 * calibration, which ran in the instant after programming while the probe was
 * still powering the domain.
 *
 * TIM2 on the STM32L4 is a 32-bit up-counter on an 80 MHz bus with the
 * prescaler at 1, so a tick is a core cycle and it wraps at the same 53
 * seconds CYCCNT would. Identical semantics, no debugger required. */
#define TIM2_BASE    0x40000000u
#define TIM2_CR1     REG(TIM2_BASE + 0x00)
#define TIM2_EGR     REG(TIM2_BASE + 0x14)
#define TIM2_CNT     REG(TIM2_BASE + 0x24)
#define TIM2_PSC     REG(TIM2_BASE + 0x28)
#define TIM2_ARR     REG(TIM2_BASE + 0x2C)

#define SYSCLK_HZ    80000000u
#define BAUD         115200u

static void clock_init(void) {
  /* Flash latency must lead the frequency, never follow it: four wait states
     for 80 MHz at VCORE range 1, which is the reset default. */
  FLASH_ACR = (FLASH_ACR & ~7u) | 4u;
  while ((FLASH_ACR & 7u) != 4u) { }

  /* MSI is 4 MHz out of reset. PLLM=1, PLLN=40, PLLR=2 gives
     4 MHz / 1 * 40 / 2 = 80 MHz, with a 160 MHz VCO. */
  RCC_PLLCFGR = 1u            /* PLLSRC = MSI   */
              | (0u << 4)     /* PLLM = /1      */
              | (40u << 8)    /* PLLN = 40      */
              | (1u << 24)    /* PLLREN         */
              | (0u << 25);   /* PLLR = /2      */
  RCC_CR |= (1u << 24);                       /* PLLON  */
  while (!(RCC_CR & (1u << 25))) { }          /* PLLRDY */

  RCC_CFGR = (RCC_CFGR & ~3u) | 3u;           /* SYSCLK = PLL */
  while (((RCC_CFGR >> 2) & 3u) != 3u) { }
}

static void uart_init(void) {
  RCC_AHB2ENR |= (1u << 1);                   /* GPIOB */
  RCC_APB2ENR |= (1u << 14);                  /* USART1 */

  /* PB6, PB7 -> AF7 */
  GPIO_MODER(GPIOB_BASE) = (GPIO_MODER(GPIOB_BASE) & ~((3u << 12) | (3u << 14)))
                         | (2u << 12) | (2u << 14);
  GPIO_AFRL(GPIOB_BASE)  = (GPIO_AFRL(GPIOB_BASE) & ~((0xFu << 24) | (0xFu << 28)))
                         | (7u << 24) | (7u << 28);

  /* USART1 is on an 80 MHz bus clock; every prescaler is /1. */
  USART_BRR(USART1_BASE) = SYSCLK_HZ / BAUD;
  USART_CR1(USART1_BASE) = (1u << 3) | (1u << 0);   /* TE | UE */
}

/* LED2 (green) on PB14: a liveness signal that survives a UART wired
   somewhere this file did not guess. */
static void led_init(void) {
  GPIO_MODER(GPIOB_BASE) = (GPIO_MODER(GPIOB_BASE) & ~(3u << 28)) | (1u << 28);
}

void board_led(int on) {
  GPIO_BSRR(GPIOB_BASE) = on ? (1u << 14) : (1u << 30);
}

static void cycles_init(void) {
  /* Enabling a peripheral clock does not make the peripheral writable on the
     next instruction: the RCC write is posted, and register writes issued
     before it lands are dropped. ST's own workaround is to read the enable
     register back, which stalls until the write has retired.
     Skipping this cost an afternoon — the timer came up dead, every
     measurement read zero, and it started working several seconds later for
     reasons that looked like anything but a missing read-back. */
  RCC_APB1ENR1 |= (1u << 0);            /* TIM2EN */
  (void)RCC_APB1ENR1;

  TIM2_CR1 = 0;
  TIM2_PSC = 0;                         /* one tick per bus cycle */
  TIM2_ARR = 0xFFFFFFFFu;
  TIM2_EGR = 1u;                        /* UG: load PSC and ARR now */
  TIM2_CR1 = 1u;                        /* CEN */

  /* And then confirm it, because the failure mode above is silent and a
     benchmark reporting zero cycles is worse than one that refuses to run.
     If the counter is not advancing there is nothing useful to measure. */
  for (;;) {
    uint32_t a = TIM2_CNT;
    for (volatile int i = 0; i < 64; i++) { }
    if (TIM2_CNT != a) return;
    TIM2_CR1 = 1u;
  }
}

void board_init(void) {
  clock_init();
  uart_init();
  led_init();
  cycles_init();
}

void board_putc(char c) {
  while (!(USART_ISR(USART1_BASE) & (1u << 7))) { }   /* TXE */
  USART_TDR(USART1_BASE) = (uint32_t)(uint8_t)c;
}

uint32_t board_cycles(void) { return TIM2_CNT; }

uint32_t board_sysclk(void) { return SYSCLK_HZ; }

/* The experiment. STM32L4 has an 8-line data cache and a 32-line instruction
   cache in front of flash, plus a prefetch buffer; weights are read through
   all three. Whether four interleaved weight streams fit in eight data-cache
   lines is not a question to settle by argument, and this is the knob that
   settles it. ST's sequence requires a cache to be reset while disabled. */
void board_flash_cache(int prefetch, int icache, int dcache) {
  uint32_t acr = FLASH_ACR;
  acr &= ~((1u << 8) | (1u << 9) | (1u << 10));
  FLASH_ACR = acr;
  FLASH_ACR = acr | (1u << 11) | (1u << 12);          /* ICRST | DCRST */
  FLASH_ACR = acr;
  if (prefetch) acr |= (1u << 8);
  if (icache)   acr |= (1u << 9);
  if (dcache)   acr |= (1u << 10);
  FLASH_ACR = acr;
}

uint32_t board_flash_acr(void) { return FLASH_ACR; }
