/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * ESP32-C3-DevKitM-1 board support: watchdogs off, a known CPU clock, the
 * console, the cycle counter, and the cache and MMU that make flash readable
 * at all.
 *
 * Registers are written by address rather than through a vendor HAL, as on the
 * two Cortex-M ports. The addresses and bit positions are from the ESP32-C3
 * TRM and from ESP-IDF v5.0's own register headers, and each block below says
 * which register file it came from so that a reader can check it against one.
 *
 * The boot ROM is the exception, and it is one on purpose. Three things here
 * — suspending the cache, invalidating it, and enabling it — are sequences of
 * undocumented register writes whose only published form is the routine in the
 * part's own mask ROM. Those are called at their ROM addresses, taken from
 * esp-idf's esp32c3.rom.ld. Reimplementing them from a guess would not make
 * this file more independent of the vendor, only less checkable. Everything
 * else, including the MMU table this port actually depends on, is written
 * here in the open.
 */

#include <stdint.h>

#define REG(a) (*(volatile uint32_t *)(a))

/* ~~ register files ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* uart_reg.h */
#define UART0_BASE      0x60000000u
#define UART_FIFO       REG(UART0_BASE + 0x00)
#define UART_CLKDIV     REG(UART0_BASE + 0x14)
#define UART_STATUS     REG(UART0_BASE + 0x1C)
#define UART_TXFIFO_CNT_S 16
#define UART_TXFIFO_CNT_V 0x3FFu

/* rtc_cntl_reg.h */
#define RTC_BASE        0x60008000u
#define RTC_WDTCONFIG0  REG(RTC_BASE + 0x0090)
#define RTC_WDTWPROTECT REG(RTC_BASE + 0x00A8)
#define RTC_SWD_CONF    REG(RTC_BASE + 0x00AC)
#define RTC_SWD_WPROTECT REG(RTC_BASE + 0x00B0)
#define RTC_WDT_WKEY    0x50D83AA1u
#define RTC_SWD_WKEY    0x8F1D312Au
#define RTC_SWD_AUTO_FEED_EN (1u << 31)

/* timer_group_reg.h; TIMG0 and TIMG1 each have a watchdog */
#define TIMG0_BASE      0x6001F000u
#define TIMG1_BASE      0x60020000u
#define TIMG_WDTCONFIG0(b)  REG((b) + 0x48)
#define TIMG_WDTWPROTECT(b) REG((b) + 0x64)
#define TIMG_WDT_WKEY   0x50D83AA1u

/* systimer_reg.h. Counts at a fixed 16 MHz off the crystal, whatever the core
   is doing, which is what makes it a usable second opinion on the core clock. */
#define SYSTIMER_BASE   0x60023000u
#define SYSTIMER_CONF   REG(SYSTIMER_BASE + 0x00)
#define SYSTIMER_U0_OP  REG(SYSTIMER_BASE + 0x04)
#define SYSTIMER_U0_HI  REG(SYSTIMER_BASE + 0x40)
#define SYSTIMER_U0_LO  REG(SYSTIMER_BASE + 0x44)
#define SYSTIMER_U0_VALUE_VALID (1u << 29)
#define SYSTIMER_U0_UPDATE      (1u << 30)
#define SYSTIMER_U0_WORK_EN     (1u << 30)
#define SYSTIMER_CLK_EN         (1u << 31)
#define SYSTIMER_HZ     16000000u

/* system_reg.h */
#define SYSTEM_BASE     0x600C0000u
#define SYSTEM_CPU_PER_CONF REG(SYSTEM_BASE + 0x008)
#define SYSTEM_SYSCLK_CONF  REG(SYSTEM_BASE + 0x058)
#define SYSTEM_CPUPERIOD_SEL_S 0
#define SYSTEM_PLL_FREQ_SEL    (1u << 2)
#define SYSTEM_SOC_CLK_SEL_S   10

/* extmem_reg.h — the cache in front of flash */
#define EXTMEM_BASE     0x600C4000u
#define EXTMEM_ICACHE_CTRL   REG(EXTMEM_BASE + 0x000)
#define EXTMEM_ICACHE_CTRL1  REG(EXTMEM_BASE + 0x004)
#define EXTMEM_ICACHE_SHUT_IBUS (1u << 0)
#define EXTMEM_ICACHE_SHUT_DBUS (1u << 1)

/* reg_base.h / ext_mem_defs.h — the flash MMU. 128 entries of one 64 KB page
   each, shared between the instruction and data windows: an entry's index is
   taken from the low 23 bits of the address, so 0x3C00_0000 + n and
   0x4200_0000 + n are the same entry. An entry holds a flash page number, or
   bit 8 to mean "not mapped". */
#define MMU_TABLE       ((volatile uint32_t *)0x600C5000u)
#define MMU_ENTRIES     128
#define MMU_INVALID     (1u << 8)
#define MMU_PAGE_SIZE   0x10000u
#define FLASH_SIZE      0x400000u          /* 4 MB, embedded on this module */

/* Boot ROM entry points, from esp-idf v5.0 components/esp_rom/esp32c3/ld/
   esp32c3.rom.ld. See the note at the top of this file for why these three
   and no others. */
#define rom_Cache_Disable_ICache() \
  ((void (*)(void))0x4000051cu)()
#define rom_Cache_Enable_ICache(autoload) \
  ((void (*)(uint32_t))0x40000520u)(autoload)
#define rom_Cache_Invalidate_ICache_All() \
  ((void (*)(void))0x400004d8u)()

/* ~~ watchdogs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * The boot ROM hands the image a running RTC watchdog and a running "super"
 * watchdog, on the assumption that whatever it started is a second-stage
 * bootloader that will shortly say so. This firmware measures a convolution
 * that takes the better part of a second and then says nothing for a minute,
 * which from a watchdog's point of view is indistinguishable from a hang.
 *
 * Feeding them instead of stopping them would be worse: the feed would land
 * inside a measured region, and the whole design of this benchmark is that
 * nothing does. */
static void watchdogs_off(void) {
  RTC_WDTWPROTECT = RTC_WDT_WKEY;
  RTC_WDTCONFIG0 = 0;
  RTC_WDTWPROTECT = 0;

  RTC_SWD_WPROTECT = RTC_SWD_WKEY;
  RTC_SWD_CONF |= RTC_SWD_AUTO_FEED_EN;   /* it cannot be stopped, only fed */
  RTC_SWD_WPROTECT = 0;

  TIMG_WDTWPROTECT(TIMG0_BASE) = TIMG_WDT_WKEY;
  TIMG_WDTCONFIG0(TIMG0_BASE) = 0;
  TIMG_WDTWPROTECT(TIMG0_BASE) = 0;

  TIMG_WDTWPROTECT(TIMG1_BASE) = TIMG_WDT_WKEY;
  TIMG_WDTCONFIG0(TIMG1_BASE) = 0;
  TIMG_WDTWPROTECT(TIMG1_BASE) = 0;
}

/* ~~ time base ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * The counter is PCCR, one of three custom machine-mode CSRs this core adds
 * for performance monitoring: PCER selects what is counted, PCMR runs it, and
 * PCCR is the 32-bit count. Event 1 is core cycles.
 *
 * `mcycle` is the obvious answer and is not one: reading it traps as an
 * illegal instruction — mcause 2, mtval the `csrr` itself — because this core
 * does not implement the standard counters any more than it implements `mie`.
 * The vendor's own code reads PCCR for the same reason.
 *
 * cycles_init below confirms the counter advances before anything is timed,
 * exactly as the STM32 port does with TIM2. A benchmark whose counter reads
 * zero reports every kernel as infinitely fast. */
#define CSR_PCER 0x7e0
#define CSR_PCMR 0x7e1
#define CSR_PCCR 0x7e2

static uint32_t cpu_hz;

uint32_t board_cycles(void) {
  uint32_t v;
  __asm__ volatile("csrr %0, %1" : "=r"(v) : "i"(CSR_PCCR));
  return v;
}

static uint64_t systimer_now(void) {
  SYSTIMER_U0_OP = SYSTIMER_U0_UPDATE;
  while (!(SYSTIMER_U0_OP & SYSTIMER_U0_VALUE_VALID)) { }
  uint32_t hi = SYSTIMER_U0_HI;
  uint32_t lo = SYSTIMER_U0_LO;
  return ((uint64_t)hi << 32) | lo;
}

static void cycles_init(void) {
  SYSTIMER_CONF |= SYSTIMER_CLK_EN | SYSTIMER_U0_WORK_EN;

  __asm__ volatile("csrw %0, %1" :: "i"(CSR_PCER), "r"(1u));   /* cycles */
  __asm__ volatile("csrw %0, %1" :: "i"(CSR_PCMR), "r"(1u));   /* count   */
  __asm__ volatile("csrw %0, zero" :: "i"(CSR_PCCR));

  for (;;) {
    uint32_t a = board_cycles();
    for (volatile int i = 0; i < 64; i++) { }
    if (board_cycles() != a) return;
    __asm__ volatile("csrw %0, %1" :: "i"(CSR_PCMR), "r"(1u));
  }
}

/* What the core clock actually is, rather than what this file believes it set.
 * Both are worth having: clock_init below asks for 160 MHz, and this counts
 * core cycles against the crystal-derived system timer to find out whether it
 * got them. The benchmark reports the measured number, so a millisecond column
 * cannot silently be computed from a frequency the part is not running at. */
static uint32_t measure_cpu_hz(void) {
  const uint64_t ticks = SYSTIMER_HZ / 100;          /* 10 ms */
  uint64_t t0 = systimer_now();
  uint32_t c0 = board_cycles();
  while (systimer_now() - t0 < ticks) { }
  uint32_t c1 = board_cycles();
  uint32_t hz = (uint32_t)((uint64_t)(c1 - c0) * SYSTIMER_HZ / ticks);

  /* Rounded to the nearest megahertz, because the quantity being measured is
     not continuous: this clock is a crystal multiplied by an integer ratio, so
     the tens of hertz the count comes back with are the measurement's error
     and not the part's. Rounding keeps that error out of the millisecond
     column without discarding the check — a core that came up at 80 MHz
     because the switch above did not take still reports 80. */
  return ((hz + 500000u) / 1000000u) * 1000000u;
}

uint32_t board_sysclk(void) { return cpu_hz; }

const char *board_timebase(void) { return "pccr"; }

/* ~~ clock ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * The boot ROM hands over at 20 MHz — the 40 MHz crystal halved — which is an
 * eighth of what this part is rated for and not a number anything should be
 * benchmarked at. It has, however, already brought the 480 MHz PLL up for its
 * own use, so reaching 160 is selecting a source and a divider rather than
 * bringing up a PLL: two fields, no lock loop, nothing to wait for. That the
 * PLL really was running is not an assumption — 480/3 is exactly the 160 MHz
 * the counter measures afterwards.
 *
 * APB does not stay put across that change: it follows the core off the
 * crystal and up to 80 MHz. The console survives anyway because the boot ROM
 * clocks UART0 from the crystal directly rather than from APB, which is the
 * one thing that makes it safe to change the core clock with a byte in the
 * transmit FIFO. A console on APB would have come out at four times its baud
 * rate from this line onwards. */
static void clock_init(void) {
  uint32_t cfg = SYSTEM_CPU_PER_CONF;
  cfg |= SYSTEM_PLL_FREQ_SEL;                        /* 480 MHz PLL */
  cfg = (cfg & ~3u) | (1u << SYSTEM_CPUPERIOD_SEL_S); /* /3 -> 160 MHz */
  SYSTEM_CPU_PER_CONF = cfg;

  uint32_t sys = SYSTEM_SYSCLK_CONF;
  sys = (sys & ~(3u << SYSTEM_SOC_CLK_SEL_S)) | (1u << SYSTEM_SOC_CLK_SEL_S);
  SYSTEM_SYSCLK_CONF = sys;
}

/* ~~ console ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * UART0 on GPIO20/21, into the CP2102N bridge the DevKitM-1 carries, already
 * configured by the boot ROM for its own log at 115200 baud. There is nothing
 * to initialise: this port inherits a working console the way the SAMD51 port
 * inherits nothing at all and has to implement USB to get one.
 *
 * The device this talks to is a bridge chip with its own buffer, so, as on the
 * STM32, there is nothing to service between characters. */
void board_poll(void) { }

void board_putc(char c) {
  while (((UART_STATUS >> UART_TXFIFO_CNT_S) & UART_TXFIFO_CNT_V) >= 126u) { }
  UART_FIFO = (uint32_t)(uint8_t)c;
}

/* The DevKitM-1's only LED is an addressable RGB part on GPIO8, which needs a
   bit-banged or RMT-driven protocol to say anything at all. The benchmark
   treats the liveness signal as optional for exactly this case. */
void board_led(int on) { (void)on; }

/* ~~ flash ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * The core cannot address flash. It addresses two 8 MB windows — 0x3C00_0000
 * for data, 0x4200_0000 for instructions — which a 128-entry MMU translates
 * into 64 KB flash pages, through a 16 KB cache shared by both. Nothing is
 * mapped when this image starts, so both the kernels and the weights are
 * unreachable until the table below is written.
 *
 * The mapping is the identity: entry n covers flash page n, so a mapped
 * address is its own flash offset in the low 23 bits and the linker script can
 * simply name the offsets it wants. Mapping the whole 4 MB rather than the two
 * regions in use costs 128 stores and removes the build-time coupling that the
 * alternative would need — the linker would have to agree with the flasher
 * about page numbers, and disagreeing silently produces a model that reads
 * plausible-looking rubbish out of the wrong part of flash.
 *
 * A mapping this port gets wrong does not go unnoticed: the benchmark
 * checksums the model's output and the harness compares it against the other
 * boards', where the same weights are addressed directly. */
static void flash_map(void) {
  rom_Cache_Disable_ICache();

  for (uint32_t i = 0; i < MMU_ENTRIES; i++)
    MMU_TABLE[i] = (i < FLASH_SIZE / MMU_PAGE_SIZE) ? i : MMU_INVALID;

  rom_Cache_Invalidate_ICache_All();
  rom_Cache_Enable_ICache(0);

  /* Both buses are shut out of the cache after reset; the mapping is worth
     nothing until they are let in. */
  EXTMEM_ICACHE_CTRL1 &= ~(EXTMEM_ICACHE_SHUT_IBUS | EXTMEM_ICACHE_SHUT_DBUS);
}

void board_init(void) {
  watchdogs_off();
  clock_init();
  cycles_init();

  /* Before this line the image can only run what is in SRAM, and that is a
     sharper constraint than it looks: it rules out the compiler's own helper
     routines as well as this project's code. Measuring the core clock ahead of
     the mapping hung the board during bring-up for exactly that reason — the
     measurement divides a 64-bit product, libgcc's division helper is linked
     with everything else into flash, and calling it was a jump into an address
     space that did not exist yet. Anything added above here has to be 32-bit
     arithmetic and nothing else. */
  flash_map();

  cpu_hz = measure_cpu_hz();
}

/* ~~ the sweep ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * One row, and the reason is structural rather than an omission. On both
 * Cortex-M parts the accelerators in front of flash are an optimisation: turn
 * them off and the core still reads flash, only slowly, which is what makes
 * "everything off" a measurable row and the honest number for a part without
 * them. Here the cache is not in front of the path to flash, it *is* the path
 * to flash — the MMU translates into it and nothing addresses flash around it.
 * Disabling it does not produce a slow read, it produces no read at all, and
 * the kernels are being fetched through it as well.
 *
 * So this board reports the one configuration it has rather than pretending to
 * a knob it does not own, and the benchmark asks how many rows there are for
 * precisely this reason. */
int board_cache_configs(void) { return 1; }

const char *board_cache_label(int i) { (void)i; return "icache=16K"; }

void board_cache_select(int i) { (void)i; }

uint32_t board_cache_state(void) { return EXTMEM_ICACHE_CTRL; }
