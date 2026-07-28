/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * ATSAMD51G19A board support (Adafruit ItsyBitsy M4 Express): clock, console,
 * cycle counter, and the caches the benchmark treats as an experimental
 * variable. Registers are written by address rather than through a vendor HAL,
 * as on the other board here; every line is checkable against the SAM D5x/E5x
 * family datasheet.
 *
 * Two things differ from the STM32 port in ways worth stating up front.
 *
 * There is no debug probe on this board and no USB-serial bridge behind it.
 * The part's own USB peripheral is the only route off it, so the console is a
 * CDC endpoint this firmware serves itself — see usb_cdc.c. That also supplies
 * the way back into the bootloader, which is how the harness reflashes.
 *
 * And the clock is recovered from USB rather than from a crystal. DFLL48M runs
 * closed-loop against the host's 1 kHz start-of-frame, DPLL0 multiplies it to
 * 120 MHz, and the core runs from that. This is not a convenience: it is what
 * makes the reported milliseconds mean anything, because an open-loop DFLL is
 * a percent or so off and every ms in the table is cycles divided by the
 * number `board_sysclk` returns.
 */

#include <stdint.h>

#define REG8(a)  (*(volatile uint8_t  *)(a))
#define REG16(a) (*(volatile uint16_t *)(a))
#define REG32(a) (*(volatile uint32_t *)(a))

/* NVMCTRL. CTRLA is 16-bit: RWS[11:8] are the flash wait states, and
   CACHEDIS0/1 (bits 14, 15) switch off the read buffers in front of the two
   AHB ports into flash. Both halves matter here — the wait states because
   they are a floor under every weight fetch, the buffers because they are
   half the experiment. */
#define NVMCTRL_BASE  0x41004000u
#define NVMCTRL_CTRLA REG16(NVMCTRL_BASE + 0x00)

/* CMCC: the 4 KB unified cache in front of the AHB, and the larger half of
   the experiment. Enabling it needs the cache idle; invalidating it needs the
   cache disabled. */
#define CMCC_BASE    0x41006000u
#define CMCC_CTRL    REG32(CMCC_BASE + 0x08)
#define CMCC_SR      REG32(CMCC_BASE + 0x0C)
#define CMCC_MAINT0  REG32(CMCC_BASE + 0x20)

/* GCLK */
#define GCLK_BASE       0x40001C00u
#define GCLK_CTRLA      REG8 (GCLK_BASE + 0x00)
#define GCLK_SYNCBUSY   REG32(GCLK_BASE + 0x04)
#define GCLK_GENCTRL(n) REG32(GCLK_BASE + 0x20 + (n) * 4u)
#define GCLK_PCHCTRL(n) REG32(GCLK_BASE + 0x80 + (n) * 4u)
#define GCLK_SYNC_GEN(n) (1u << (2 + (n)))

#define GCLK_SRC_OSCULP32K 4u
#define GCLK_SRC_DFLL      6u
#define GCLK_SRC_DPLL0     7u
#define GCLK_GENEN         (1u << 8)
#define GCLK_IDC           (1u << 9)
#define GCLK_CHEN          (1u << 6)

/* Peripheral channels, from the PCHCTRL mapping table. */
#define PCH_FDPLL0  1u
#define PCH_TC0     9u
#define PCH_USB    10u

/* OSCCTRL. The three-byte gaps after the 8-bit DFLLCTRLA, DFLLCTRLB and
   DFLLSYNC are not decoration: get them wrong and every DFLL and DPLL write
   below lands one register late, which configures nothing and reports no
   error. The part still runs — the core clock is whatever the oscillator does
   untouched — and USB, which needs 48 MHz to a quarter of a percent, does not.
   These offsets are the ones in the family header, checked rather than
   counted. */
#define OSCCTRL_BASE     0x40001000u
#define OSCCTRL_STATUS   REG32(OSCCTRL_BASE + 0x10)
#define OSCCTRL_DFLLCTRLA REG8 (OSCCTRL_BASE + 0x1C)
#define OSCCTRL_DFLLCTRLB REG8 (OSCCTRL_BASE + 0x20)
#define OSCCTRL_DFLLVAL  REG32(OSCCTRL_BASE + 0x24)
#define OSCCTRL_DFLLMUL  REG32(OSCCTRL_BASE + 0x28)
#define OSCCTRL_DFLLSYNC REG8 (OSCCTRL_BASE + 0x2C)
#define OSCCTRL_DPLL0CTRLA    REG8 (OSCCTRL_BASE + 0x30)
#define OSCCTRL_DPLL0RATIO    REG32(OSCCTRL_BASE + 0x34)
#define OSCCTRL_DPLL0CTRLB    REG32(OSCCTRL_BASE + 0x38)
#define OSCCTRL_DPLL0SYNCBUSY REG32(OSCCTRL_BASE + 0x3C)
#define OSCCTRL_DPLL0STATUS   REG32(OSCCTRL_BASE + 0x40)

#define DFLL_ENABLE     (1u << 1)
#define DFLLSYNC_ENABLE (1u << 1)
#define DFLLSYNC_CTRLB  (1u << 2)
#define DFLLSYNC_VAL    (1u << 3)
#define DFLLSYNC_MUL    (1u << 4)
#define DFLLB_USBCRM    (1u << 3)
#define DFLLB_CCDIS     (1u << 4)
#define DFLLB_WAITLOCK  (1u << 7)
#define STATUS_DFLLRDY  (1u << 8)

/* MCLK */
#define MCLK_BASE     0x40000800u
#define MCLK_AHBMASK  REG32(MCLK_BASE + 0x10)
#define MCLK_APBAMASK REG32(MCLK_BASE + 0x14)
#define MCLK_APBBMASK REG32(MCLK_BASE + 0x18)

/* PORT, in groups of 0x80: A is group 0, B is group 1. */
#define PORT_BASE          0x41008000u
#define PORT_G(g)          (PORT_BASE + (g) * 0x80u)
#define PORT_DIRSET(g)     REG32(PORT_G(g) + 0x08)
#define PORT_OUTCLR(g)     REG32(PORT_G(g) + 0x14)
#define PORT_OUTSET(g)     REG32(PORT_G(g) + 0x18)
#define PORT_OUTTGL(g)     REG32(PORT_G(g) + 0x1C)
#define PORT_PMUX(g, p)    REG8 (PORT_G(g) + 0x30 + ((p) >> 1))
#define PORT_PINCFG(g, p)  REG8 (PORT_G(g) + 0x40 + (p))

/* TC0, paired with TC1 into one 32-bit counter — the fallback time base. */
#define TC0_BASE     0x40003800u
#define TC0_CTRLA    REG32(TC0_BASE + 0x00)
#define TC0_CTRLBSET REG8 (TC0_BASE + 0x05)
#define TC0_SYNCBUSY REG32(TC0_BASE + 0x10)
#define TC0_COUNT    REG32(TC0_BASE + 0x14)

/* Cortex-M4 debug unit — the preferred time base, if it runs unattended. */
#define DEMCR      REG32(0xE000EDFCu)
#define DWT_CTRL   REG32(0xE0001000u)
#define DWT_CYCCNT REG32(0xE0001004u)

/* The red LED, D13 on the silkscreen, per the board's CircuitPython pin
   definition. Nothing here depends on this being the right pin: it is a
   liveness signal, and on this board the stronger one is that a serial port
   appears at all. */
#define LED_GROUP 0u
#define LED_PIN   22u

void usb_cdc_init(void);
void usb_cdc_task(void);
int  usb_cdc_configured(void);
void usb_cdc_putc(char c);
void usb_cdc_flush(void);
void usb_cdc_reboot_bootloader(void);

static uint32_t sysclk_hz = 120000000u;
static int use_dwt = 1;

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* CLOCK                                                                     */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* Every wait below is bounded. A bare-metal image with no console yet has
   exactly one way to report a clock that never came up, and it is to hang;
   coming up on a slower clock and saying so is better, since `sysclk` is in
   the record and every millisecond in the table is derived from it.
 *
 * Two limits rather than one, because the bring-up crosses a factor of 3600
 * in clock rate part way through and no single number is right on both sides
 * of that: a bound that is generous at 48 MHz is minutes of an apparently
 * dead board at 32 kHz. */
#define SPIN_SLOW  2000u        /* while parked on the 32 kHz oscillator */
#define SPIN_LIMIT 500000u      /* once running at 48 MHz or better      */
#define SPIN_BOUNDED(lim, cond) do {                \
    uint32_t _n = (lim);                            \
    while (!(cond) && --_n) { }                     \
  } while (0)
#define SPIN_UNTIL(cond) SPIN_BOUNDED(SPIN_LIMIT, cond)

static void clock_init(void) {
  /* Wait states lead the frequency, never follow it. Five is the datasheet
     figure for 120 MHz at 3.3 V, and it is set before anything speeds up. */
  NVMCTRL_CTRLA = (uint16_t)((NVMCTRL_CTRLA & ~(0xFu << 8)) | (5u << 8));

  /* Park the core on the always-available 32 kHz oscillator first.
   *
   * The DFLL has to be taken down and rebuilt below, and out of reset — or
   * out of the bootloader, which is the case that actually applies — GCLK0 is
   * sourced from it. Disabling the oscillator that is clocking the core while
   * it is clocking the core is the kind of thing that works on the bench and
   * then does not. This costs a few milliseconds of running at 32 kHz and
   * removes the whole question of what state the bootloader left behind. */
  GCLK_CTRLA = 1u;                                   /* SWRST */
  SPIN_BOUNDED(SPIN_SLOW, !(GCLK_SYNCBUSY & 1u));
  GCLK_GENCTRL(0) = GCLK_SRC_OSCULP32K | GCLK_GENEN;
  SPIN_BOUNDED(SPIN_SLOW, !(GCLK_SYNCBUSY & GCLK_SYNC_GEN(0)));

  /* Bring the DFLL up open loop, which is what it does out of its own factory
     trim, and get off the 32 kHz oscillator as soon as it is running. Only
     these four writes happen at 32 kHz; the closed-loop part below wants a
     timeout measured in milliseconds, and a millisecond here is 33 cycles. */
  OSCCTRL_DFLLCTRLA = 0;
  SPIN_BOUNDED(SPIN_SLOW, !(OSCCTRL_DFLLSYNC & DFLLSYNC_ENABLE));
  OSCCTRL_DFLLMUL = (1u << 26) | (1u << 16) | 0u;    /* CSTEP | FSTEP | MUL */
  SPIN_BOUNDED(SPIN_SLOW, !(OSCCTRL_DFLLSYNC & DFLLSYNC_MUL));
  OSCCTRL_DFLLCTRLB = 0;
  SPIN_BOUNDED(SPIN_SLOW, !(OSCCTRL_DFLLSYNC & DFLLSYNC_CTRLB));
  OSCCTRL_DFLLCTRLA = DFLL_ENABLE;
  SPIN_BOUNDED(SPIN_SLOW, !(OSCCTRL_DFLLSYNC & DFLLSYNC_ENABLE));
  OSCCTRL_DFLLVAL = OSCCTRL_DFLLVAL;                 /* re-latch the factory trim */
  SPIN_BOUNDED(SPIN_SLOW, !(OSCCTRL_DFLLSYNC & DFLLSYNC_VAL));

  GCLK_GENCTRL(0) = GCLK_SRC_DFLL | GCLK_GENEN | GCLK_IDC | (1u << 16);
  SPIN_BOUNDED(SPIN_SLOW, !(GCLK_SYNCBUSY & GCLK_SYNC_GEN(0)));
  sysclk_hz = 48000000u;

  /* Now ask for closed loop on the USB host's start-of-frame. The multiplier
     is forced by the hardware in this mode; CSTEP and FSTEP above are the
     coarse and fine step limits the part's reference sequence uses.
   *
   * WAITLOCK holds DFLLRDY until the loop has locked, and with no host
   * attached — which is the case here, since USB has not been enabled yet —
   * there are no frames to lock to and it never will. That is why the wait is
   * bounded and falling out of it is not an error: the oscillator free-runs
   * near 48 MHz on its factory trim, which is close enough to enumerate, and
   * the lock arrives a millisecond after the host's first frame does. */
  OSCCTRL_DFLLCTRLB = DFLLB_WAITLOCK | DFLLB_CCDIS | DFLLB_USBCRM;
  SPIN_UNTIL(!(OSCCTRL_DFLLSYNC & DFLLSYNC_CTRLB));
  SPIN_UNTIL(OSCCTRL_STATUS & STATUS_DFLLRDY);

  /* GCLK1 = 48 MHz for USB, GCLK5 = 2 MHz as the DPLL reference. Dividing
     down to 2 MHz rather than feeding the DPLL 48 MHz directly is what the
     part's own reference sequence does: the phase detector wants a low
     reference and the loop filter is tuned for one. */
  GCLK_GENCTRL(1) = GCLK_SRC_DFLL | GCLK_GENEN | GCLK_IDC | (1u << 16);
  SPIN_UNTIL(!(GCLK_SYNCBUSY & GCLK_SYNC_GEN(1)));
  GCLK_GENCTRL(5) = GCLK_SRC_DFLL | GCLK_GENEN | GCLK_IDC | (24u << 16);
  SPIN_UNTIL(!(GCLK_SYNCBUSY & GCLK_SYNC_GEN(5)));

  /* DPLL0: 2 MHz * (59 + 1) = 120 MHz. LBYPASS skips the lock detector, which
     the reference sequence does for an integer ratio off a clean reference. */
  GCLK_PCHCTRL(PCH_FDPLL0) = 5u | GCLK_CHEN;
  OSCCTRL_DPLL0RATIO = 59u;
  SPIN_UNTIL(!(OSCCTRL_DPLL0SYNCBUSY & (1u << 2)));
  OSCCTRL_DPLL0CTRLB = (0u << 5) | (1u << 11);       /* REFCLK = GCLK | LBYPASS */
  OSCCTRL_DPLL0CTRLA = (1u << 1);                    /* ENABLE */
  SPIN_UNTIL(!(OSCCTRL_DPLL0SYNCBUSY & (1u << 1)));

  {
    uint32_t n = SPIN_LIMIT;
    while (!(OSCCTRL_DPLL0STATUS & 0x3u) && --n) { }  /* LOCK | CLKRDY */
    if (n) {
      GCLK_GENCTRL(0) = GCLK_SRC_DPLL0 | GCLK_GENEN | GCLK_IDC | (1u << 16);
      sysclk_hz = 120000000u;
    } else {
      /* No lock. Run from the DFLL directly and report 48 MHz, so the cycle
         counts stay exact and the milliseconds derived from them stay true. */
      GCLK_GENCTRL(0) = GCLK_SRC_DFLL | GCLK_GENEN | GCLK_IDC | (1u << 16);
      sysclk_hz = 48000000u;
    }
    SPIN_UNTIL(!(GCLK_SYNCBUSY & GCLK_SYNC_GEN(0)));
  }

  /* Clocks for the peripherals this file and usb_cdc.c touch. Most of these
     bits are set out of reset; setting them again is free and does not depend
     on that staying true. */
  MCLK_AHBMASK  |= (1u << 8) | (1u << 10);           /* CMCC | USB */
  MCLK_APBAMASK |= (1u << 14) | (1u << 15);          /* TC0 | TC1 */
  MCLK_APBBMASK |= (1u << 0);                        /* USB */
  GCLK_PCHCTRL(PCH_USB) = 1u | GCLK_CHEN;            /* 48 MHz */
  GCLK_PCHCTRL(PCH_TC0) = 0u | GCLK_CHEN;            /* core clock */
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* LED                                                                       */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

static void led_init(void) {
  PORT_PINCFG(LED_GROUP, LED_PIN) = 0;               /* plain output, no mux */
  PORT_DIRSET(LED_GROUP) = (1u << LED_PIN);
}

void board_led(int on) {
  if (on) PORT_OUTSET(LED_GROUP) = (1u << LED_PIN);
  else    PORT_OUTCLR(LED_GROUP) = (1u << LED_PIN);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* CYCLE COUNTER                                                             */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* DWT's CYCCNT is the natural answer and on the STM32 in this directory it is
 * the wrong one: there the counter lives in a debug power domain that is off
 * whenever a probe is not attached, so it reads zero for exactly the runs
 * anyone cares about. That is a fact about that part, not about Cortex-M4, and
 * this board has no probe attached to it ever — so the counter is tried,
 * checked against a delay that must advance it, and used only if it does.
 *
 * The fallback is TC0 and TC1 chained into one 32-bit counter off the core
 * clock, which is the same construction the STM32 port uses and has identical
 * semantics: one tick per core cycle, wrapping at 2^32. Its one cost is that
 * reading COUNT on this part needs a READSYNC command and a synchroniser
 * round trip first, which is why it is the fallback and not the default —
 * that read is tens of cycles where DWT's is one, and it lands on the fastest
 * operators. `calibrateOverhead` in bench.nim.in subtracts whichever it is.
 */
static void tc_init(void) {
  TC0_CTRLA = 1u;                                    /* SWRST */
  SPIN_UNTIL(!(TC0_SYNCBUSY & 1u));
  TC0_CTRLA = (2u << 2);                             /* MODE = COUNT32 */
  TC0_CTRLA |= (1u << 1);                            /* ENABLE */
  SPIN_UNTIL(!(TC0_SYNCBUSY & (1u << 1)));
}

static uint32_t tc_read(void) {
  TC0_CTRLBSET = (4u << 5);                          /* CMD = READSYNC */
  SPIN_UNTIL(!(TC0_SYNCBUSY & (1u << 4)));
  return TC0_COUNT;
}

static int counter_advances(uint32_t (*read)(void)) {
  uint32_t a = read();
  for (volatile int i = 0; i < 256; i++) { }
  return read() != a;
}

static uint32_t dwt_read(void) { return DWT_CYCCNT; }

static void cycles_init(void) {
  DEMCR |= (1u << 24);                               /* TRCENA */
  DWT_CYCCNT = 0;
  DWT_CTRL |= 1u;                                    /* CYCCNTENA */
  if (counter_advances(dwt_read)) { use_dwt = 1; return; }

  use_dwt = 0;
  tc_init();
  /* And then confirm that one too, because a benchmark reporting zero cycles
     is worse than one that refuses to run. If neither counter advances there
     is nothing here worth measuring, and hanging says so. */
  while (!counter_advances(tc_read)) {
    tc_init();
  }
}

uint32_t board_cycles(void) { return use_dwt ? DWT_CYCCNT : tc_read(); }

/* Which of the two answered, in the record, because "cycles" means something
   slightly different for each and the reader should not have to guess. */
const char *board_timebase(void) { return use_dwt ? "dwt-cyccnt" : "tc0-count32"; }

uint32_t board_sysclk(void) { return sysclk_hz; }

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* THE EXPERIMENT                                                            */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* Two independent things sit between a weight array and the core: a 4 KB
 * unified cache (CMCC), and a pair of read buffers inside the flash
 * controller, one per AHB port. They are different mechanisms at different
 * sizes, so the sweep takes them off in that order and the last row has
 * neither — which is the row the per-operator table is reported from, on this
 * board for the same reason as the other one.
 *
 * This is not the same set of switches as the STM32's prefetch buffer and two
 * caches, and it deliberately does not pretend to be. The labels say what the
 * part has.
 */
static void cache_apply(int cmcc, int nvm) {
  uint16_t a;

  CMCC_CTRL = 0;
  SPIN_UNTIL(!(CMCC_SR & 1u));
  CMCC_MAINT0 = 1u;                                  /* INVALL, while disabled */
  if (cmcc) CMCC_CTRL = 1u;

  a = NVMCTRL_CTRLA;
  a = (uint16_t)(a & ~((1u << 14) | (1u << 15)));    /* CACHEDIS0 | CACHEDIS1 */
  if (!nvm) a = (uint16_t)(a | (1u << 14) | (1u << 15));
  NVMCTRL_CTRLA = a;
}

int board_cache_configs(void) { return 3; }

const char *board_cache_label(int i) {
  switch (i) {
    case 0:  return "cmcc=1 nvmbuf=1";
    case 1:  return "cmcc=0 nvmbuf=1";
    default: return "cmcc=0 nvmbuf=0";
  }
}

void board_cache_select(int i) {
  switch (i) {
    case 0:  cache_apply(1, 1); break;
    case 1:  cache_apply(0, 1); break;
    default: cache_apply(0, 0); break;
  }
}

uint32_t board_cache_state(void) {
  return (uint32_t)NVMCTRL_CTRLA | ((CMCC_SR & 1u) << 16);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* CONSOLE                                                                   */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* The console is a USB endpoint this firmware serves itself, so someone has
   to turn the crank; `bench.nim.in` calls this between measured regions. */
void board_poll(void) { usb_cdc_task(); }

void board_putc(char c) {
  usb_cdc_putc(c);
  /* Flush per line rather than per packet. A record is read by a host that
     wants to see it arrive, and a line is well under one 64-byte packet. */
  if (c == '\n') usb_cdc_flush();
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void board_init(void) {
  clock_init();
  led_init();
  usb_cdc_init();

  /* Wait to be enumerated before doing anything else, blinking while not.
   *
   * Two reasons, and the second is the real one. The console does not exist
   * until a host has configured it, so output before that goes nowhere. And
   * the core clock is only as accurate as the DFLL's lock, which comes from
   * the host's frames — so a measurement taken before enumeration would be
   * against a clock that is a percent or so away from the one the record
   * claims. Waiting costs a second and removes both.
   *
   * And if it never happens, hand the board back to the bootloader.
   *
   * That is not a nicety. This part is programmed over USB and nothing else:
   * an image that fails to enumerate is an image that cannot be replaced,
   * except by someone physically double-tapping the reset button. Giving up
   * after twenty-odd seconds and asking for the bootloader means the worst a
   * broken build can do is cost a reflash, which is the difference between a
   * board that can be iterated on remotely and one that cannot. */
  {
    uint32_t spin = 0, blinks = 0;
    while (!usb_cdc_configured()) {
      usb_cdc_task();
      if (++spin >= 200000u) {
        spin = 0;
        PORT_OUTTGL(LED_GROUP) = (1u << LED_PIN);
        if (++blinks >= 600u) usb_cdc_reboot_bootloader();
      }
    }
    board_led(0);
  }

  cycles_init();
  board_cache_select(0);
}
