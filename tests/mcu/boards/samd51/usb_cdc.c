/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * A full-speed USB CDC-ACM device, polled, for the SAMD51's USB peripheral.
 * It exists because this board has no debug probe and no USB-serial bridge:
 * the part's own USB is the only way a cycle count gets off it.
 *
 * This is deliberately not a USB stack. It serves one configuration, one
 * console, and the six or so control requests enumeration actually asks for,
 * and it does all of it from a polling loop — because the one property this
 * firmware must not lose is that no interrupt fires anywhere in the image
 * while an operator is being timed. An interrupt-driven console would be less
 * code and would quietly make every measurement in the table a measurement of
 * something else.
 *
 * The consequence is that USB goes unserviced for as long as a measured
 * region runs, which is seconds. That is fine and is not luck: after
 * enumeration a CDC host sends nothing on its own, the hardware NAKs anything
 * that does arrive rather than failing it, and start-of-frame keeps the bus
 * out of suspend without software. Control requests are answered between
 * records, which is when they are asked.
 *
 * Written against the SAM D5x/E5x datasheet's USB chapter. The device
 * descriptor table lives in RAM and the peripheral reads it directly: bank 0
 * of an endpoint is its OUT bank, bank 1 its IN bank, BK0RDY means "the OUT
 * bank is full", and BK1RDY means "the IN bank has something to send". Those
 * four sentences explain most of what follows.
 */

#include <stdint.h>

#define REG8(a)  (*(volatile uint8_t  *)(a))
#define REG16(a) (*(volatile uint16_t *)(a))
#define REG32(a) (*(volatile uint32_t *)(a))

#define USB_BASE      0x41000000u
#define USB_CTRLA     REG8 (USB_BASE + 0x00)
#define USB_SYNCBUSY  REG8 (USB_BASE + 0x02)
#define USB_QOSCTRL   REG8 (USB_BASE + 0x03)
#define USB_CTRLB     REG16(USB_BASE + 0x08)
#define USB_DADD      REG8 (USB_BASE + 0x0A)
#define USB_INTFLAG   REG16(USB_BASE + 0x1C)
#define USB_DESCADD   REG32(USB_BASE + 0x24)
#define USB_PADCAL    REG16(USB_BASE + 0x28)

#define USB_EP(n, o)      (USB_BASE + 0x100u + (n) * 0x20u + (o))
#define EPCFG(n)          REG8(USB_EP(n, 0x00))
#define EPSTATUSCLR(n)    REG8(USB_EP(n, 0x04))
#define EPSTATUSSET(n)    REG8(USB_EP(n, 0x05))
#define EPSTATUS(n)       REG8(USB_EP(n, 0x06))
#define EPINTFLAG(n)      REG8(USB_EP(n, 0x07))

#define INTFLAG_EORST  (1u << 3)

#define EPST_DTGLOUT (1u << 0)
#define EPST_DTGLIN  (1u << 1)
#define EPST_STALLRQ0 (1u << 4)
#define EPST_STALLRQ1 (1u << 5)
#define EPST_BK0RDY  (1u << 6)
#define EPST_BK1RDY  (1u << 7)

#define EPINT_TRCPT0 (1u << 0)
#define EPINT_TRCPT1 (1u << 1)
#define EPINT_RXSTP  (1u << 4)

#define EPTYPE_CONTROL   1u
#define EPTYPE_BULK      3u
#define EPTYPE_INTERRUPT 4u

/* PCKSIZE.SIZE is an encoding, not a length: 3 is the 64-byte bank these
   endpoints use, and everything here is 64 bytes or smaller. */
#define PCKSIZE_64 (3u << 28)

#define EP_NOTIFY 1u   /* interrupt IN, required by the class, never used */
#define EP_DATA   2u   /* bulk IN and OUT: the console */

/* The console endpoint's own consumer: the harness reads it and nothing
   writes back, so OUT is drained and discarded. */
#define MAX_PACKET 64u

typedef struct {
  uint32_t addr;
  uint32_t pcksize;
  uint16_t extreg;
  uint8_t  status_bk;
  uint8_t  reserved[5];
} bank_t;

typedef struct { bank_t bank[2]; } ep_desc_t;

static __attribute__((aligned(4))) ep_desc_t ep_desc[3];
static __attribute__((aligned(4))) uint8_t ep0_out_buf[MAX_PACKET];
static __attribute__((aligned(4))) uint8_t ep0_in_buf[MAX_PACKET];
static __attribute__((aligned(4))) uint8_t data_out_buf[MAX_PACKET];
static __attribute__((aligned(4))) uint8_t data_in_buf[MAX_PACKET];

static uint8_t tx_len;
static uint8_t tx_busy;
static int configured;
static uint8_t pending_address;

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* DESCRIPTORS                                                               */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* 1209:0001 is pid.codes' documented identifier for exactly this — a
   one-off, non-commercial device. Borrowing the board vendor's ID for
   firmware they did not write would collide with their own products in
   every udev rule on the machine. */
static const uint8_t device_desc[] = {
  18, 0x01,              /* bLength, DEVICE                */
  0x00, 0x02,            /* bcdUSB 2.00                    */
  0x02, 0x00, 0x00,      /* class CDC, subclass 0, proto 0 */
  64,                    /* bMaxPacketSize0                */
  0x09, 0x12,            /* idVendor  0x1209               */
  0x01, 0x00,            /* idProduct 0x0001               */
  0x00, 0x01,            /* bcdDevice 1.00                 */
  1, 2, 3,               /* iManufacturer, iProduct, iSerial */
  1                      /* bNumConfigurations             */
};

#define CONFIG_TOTAL 67

static const uint8_t config_desc[CONFIG_TOTAL] = {
  /* configuration */
  9, 0x02, CONFIG_TOTAL, 0x00, 2, 1, 0, 0x80, 50,

  /* interface 0: CDC communications, ACM */
  9, 0x04, 0, 0, 1, 0x02, 0x02, 0x00, 0,
  5, 0x24, 0x00, 0x10, 0x01,          /* header, CDC 1.10        */
  5, 0x24, 0x01, 0x00, 1,             /* call management, data 1 */
  4, 0x24, 0x02, 0x02,                /* ACM: supports line coding */
  5, 0x24, 0x06, 0, 1,                /* union: control 0, sub 1 */
  7, 0x05, 0x80 | EP_NOTIFY, 0x03, 8, 0, 16,

  /* interface 1: CDC data */
  9, 0x04, 1, 0, 2, 0x0A, 0x00, 0x00, 0,
  7, 0x05, EP_DATA,        0x02, MAX_PACKET, 0, 0,
  7, 0x05, 0x80 | EP_DATA, 0x02, MAX_PACKET, 0, 0
};

static const uint8_t str_langid[] = { 4, 0x03, 0x09, 0x04 };

/* Built in place rather than stored as UTF-16 literals, which would take more
   space in this file than the code that builds them. Sized for the longest,
   which is the serial number: two bytes of header and two per character of a
   32-digit hex string. */
static uint8_t str_buf[80];

static uint32_t ascii_string_desc(const char *s) {
  uint32_t n = 0;
  while (s[n] && (2 * n + 2) < sizeof(str_buf) - 1) {
    str_buf[2 + 2 * n] = (uint8_t)s[n];
    str_buf[3 + 2 * n] = 0;
    n++;
  }
  str_buf[0] = (uint8_t)(2 + 2 * n);
  str_buf[1] = 0x03;
  return str_buf[0];
}

/* The part's 128-bit serial number, which the datasheet stores in four words
   that are not adjacent. Using it means the port's /dev/serial/by-id path is
   stable per board rather than per firmware, which is what a harness that may
   one day see two of them wants. */
static uint32_t serial_string_desc(void) {
  static const uint32_t words[4] = { 0x008061FCu, 0x00806010u,
                                     0x00806014u, 0x00806018u };
  static const char hex[] = "0123456789ABCDEF";
  uint32_t n = 0, i, j;
  for (i = 0; i < 4; i++) {
    uint32_t w = REG32(words[i]);
    for (j = 0; j < 8; j++) {
      str_buf[2 + 2 * n] = (uint8_t)hex[(w >> (28 - 4 * j)) & 0xFu];
      str_buf[3 + 2 * n] = 0;
      n++;
    }
  }
  str_buf[0] = (uint8_t)(2 + 2 * n);
  str_buf[1] = 0x03;
  return str_buf[0];
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* ENDPOINT PLUMBING                                                         */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

static void copy(uint8_t *dst, const uint8_t *src, uint32_t n) {
  uint32_t i;
  for (i = 0; i < n; i++) dst[i] = src[i];
}

static void ep0_arm_out(void) {
  ep_desc[0].bank[0].addr = (uint32_t)(uintptr_t)ep0_out_buf;
  ep_desc[0].bank[0].pcksize = PCKSIZE_64 | (MAX_PACKET << 14);
  EPINTFLAG(0) = EPINT_TRCPT0;
  EPSTATUSCLR(0) = EPST_BK0RDY;
}

/* Send one packet on EP0 and wait for the hardware to have shipped it. The
   wait is bounded by a bus reset rather than by a timer: if the host has gone
   away, EORST is the event that says so, and returning early on it is what
   lets enumeration start over cleanly. */
static int ep0_write_packet(const uint8_t *data, uint32_t n) {
  copy(ep0_in_buf, data, n);
  ep_desc[0].bank[1].addr = (uint32_t)(uintptr_t)ep0_in_buf;
  ep_desc[0].bank[1].pcksize = PCKSIZE_64 | n;
  EPINTFLAG(0) = EPINT_TRCPT1;
  EPSTATUSSET(0) = EPST_BK1RDY;
  while (!(EPINTFLAG(0) & EPINT_TRCPT1)) {
    if (USB_INTFLAG & INTFLAG_EORST) return 0;
    if (EPINTFLAG(0) & EPINT_RXSTP) return 0;   /* host abandoned this one */
  }
  EPINTFLAG(0) = EPINT_TRCPT1;
  return 1;
}

/* The data stage of a control-IN transfer, truncated to what was asked for.
   A transfer shorter than the request has to end in a short packet, so an
   exact multiple of the packet size needs an explicit empty one after it —
   otherwise the host waits for a continuation that never comes. */
static void ep0_write(const uint8_t *data, uint32_t len, uint32_t wlen) {
  uint32_t sent = 0;
  if (len > wlen) len = wlen;
  do {
    uint32_t n = (len - sent > MAX_PACKET) ? MAX_PACKET : len - sent;
    if (!ep0_write_packet(data + sent, n)) return;
    sent += n;
  } while (sent < len);
  if (len < wlen && (len % MAX_PACKET) == 0) (void)ep0_write_packet(ep0_in_buf, 0);
}

static void ep0_status_in(void) { (void)ep0_write_packet(ep0_in_buf, 0); }

static void ep0_stall(void) {
  EPSTATUSSET(0) = EPST_STALLRQ0 | EPST_STALLRQ1;
}

static void ep_configure(void) {
  uint32_t i;
  for (i = 0; i < 3; i++) {
    EPCFG(i) = 0;
    ep_desc[i].bank[0].pcksize = 0;
    ep_desc[i].bank[1].pcksize = 0;
  }

  EPCFG(0) = (uint8_t)(EPTYPE_CONTROL | (EPTYPE_CONTROL << 4));
  ep0_arm_out();

  EPCFG(EP_NOTIFY) = (uint8_t)(EPTYPE_INTERRUPT << 4);
  ep_desc[EP_NOTIFY].bank[1].addr = (uint32_t)(uintptr_t)ep0_in_buf;
  ep_desc[EP_NOTIFY].bank[1].pcksize = (0u << 28);

  EPCFG(EP_DATA) = (uint8_t)(EPTYPE_BULK | (EPTYPE_BULK << 4));
  ep_desc[EP_DATA].bank[0].addr = (uint32_t)(uintptr_t)data_out_buf;
  ep_desc[EP_DATA].bank[0].pcksize = PCKSIZE_64 | (MAX_PACKET << 14);
  ep_desc[EP_DATA].bank[1].addr = (uint32_t)(uintptr_t)data_in_buf;
  ep_desc[EP_DATA].bank[1].pcksize = PCKSIZE_64;
  EPSTATUSCLR(EP_DATA) = EPST_BK0RDY | EPST_BK1RDY;
  EPINTFLAG(EP_DATA) = EPINT_TRCPT0 | EPINT_TRCPT1;

  tx_len = 0;
  tx_busy = 0;
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* REBOOT INTO THE BOOTLOADER                                                */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* The UF2 bootloader stays resident in the first 16 KB of flash and decides
 * between itself and the application by a magic word in the last word of RAM,
 * which is what a double-tap of the reset button writes. Writing it here and
 * resetting is the same request made in software, and it is what lets the
 * harness flash the next model without anyone touching the board.
 *
 * The convention for asking is the one every Arduino-compatible bootloader
 * uses: the host opens the port at 1200 baud and drops DTR. Requiring both
 * halves — the odd baud rate *and* the transition — is what keeps a port
 * scanner that opens the console from rebooting the board mid-run.
 */
#define DBL_TAP_MAGIC 0xF01669EFu
#define DBL_TAP_PTR   REG32(0x20000000u + 192u * 1024u - 4u)
#define SCB_AIRCR     REG32(0xE000ED0Cu)

static int touch_armed;

void usb_cdc_reboot_bootloader(void) {
  DBL_TAP_PTR = DBL_TAP_MAGIC;
  __asm__ volatile ("dsb");
  SCB_AIRCR = 0x05FA0004u;                 /* VECTKEY | SYSRESETREQ */
  for (;;) { }
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* CONTROL TRANSFERS                                                         */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

/* The host's idea of the line: baud, stop bits, parity, data bits. Nothing
   here uses any of it — the endpoint is not a UART — but GET_LINE_CODING has
   to return whatever SET_LINE_CODING last set, and the baud field is how the
   reboot request arrives. */
static uint8_t line_coding[7] = { 0x00, 0xC2, 0x01, 0x00, 0, 0, 8 };

static void handle_setup(void) {
  const uint8_t bmRequestType = ep0_out_buf[0];
  const uint8_t bRequest      = ep0_out_buf[1];
  const uint16_t wValue  = (uint16_t)(ep0_out_buf[2] | (ep0_out_buf[3] << 8));
  const uint16_t wLength = (uint16_t)(ep0_out_buf[6] | (ep0_out_buf[7] << 8));

  /* Clear the completion flags along with RXSTP. The status stage of the
     *previous* transfer sets TRCPT0, and a stale one would satisfy the wait
     in SET_LINE_CODING below before its data had arrived. */
  EPINTFLAG(0) = EPINT_RXSTP | EPINT_TRCPT0 | EPINT_TRCPT1;
  EPSTATUSCLR(0) = EPST_BK0RDY;

  if ((bmRequestType & 0x60) == 0x00) {            /* standard */
    switch (bRequest) {
      case 0x00:                                   /* GET_STATUS */
        { const uint8_t z[2] = { 0, 0 }; ep0_write(z, 2, wLength); }
        return;
      case 0x01:                                   /* CLEAR_FEATURE */
      case 0x03:                                   /* SET_FEATURE */
        ep0_status_in();
        return;
      case 0x05:                                   /* SET_ADDRESS */
        /* The address takes effect only after the status stage: reply first,
           then adopt it, or the host's next token goes to a device that has
           already stopped listening on address zero. */
        pending_address = (uint8_t)(wValue & 0x7Fu);
        ep0_status_in();
        USB_DADD = (uint8_t)(pending_address | 0x80u);   /* ADDEN */
        return;
      case 0x06:                                   /* GET_DESCRIPTOR */
        switch (wValue >> 8) {
          case 1: ep0_write(device_desc, sizeof(device_desc), wLength); return;
          case 2: ep0_write(config_desc, CONFIG_TOTAL, wLength); return;
          case 3:
            switch (wValue & 0xFF) {
              case 0: ep0_write(str_langid, sizeof(str_langid), wLength); return;
              case 1: ep0_write(str_buf, ascii_string_desc("Steady"), wLength); return;
              case 2: ep0_write(str_buf, ascii_string_desc("Steady MCU benchmark"),
                                wLength); return;
              case 3: ep0_write(str_buf, serial_string_desc(), wLength); return;
              default: ep0_stall(); return;
            }
          default:
            /* DEVICE_QUALIFIER and the other high-speed descriptors: a
               full-speed-only device answers by stalling, and Linux asks. */
            ep0_stall();
            return;
        }
      case 0x08:                                   /* GET_CONFIGURATION */
        { const uint8_t c = (uint8_t)(configured ? 1 : 0);
          ep0_write(&c, 1, wLength); }
        return;
      case 0x09:                                   /* SET_CONFIGURATION */
        configured = (wValue != 0);
        ep0_status_in();
        return;
      case 0x0A:                                   /* GET_INTERFACE */
        { const uint8_t z = 0; ep0_write(&z, 1, wLength); }
        return;
      case 0x0B:                                   /* SET_INTERFACE */
        ep0_status_in();
        return;
      default:
        ep0_stall();
        return;
    }
  }

  if ((bmRequestType & 0x60) == 0x20) {            /* class: CDC */
    switch (bRequest) {
      case 0x20:                                   /* SET_LINE_CODING */
        /* The data stage is one 7-byte OUT packet. */
        ep0_arm_out();
        while (!(EPINTFLAG(0) & EPINT_TRCPT0)) {
          if (USB_INTFLAG & INTFLAG_EORST) return;
          if (EPINTFLAG(0) & EPINT_RXSTP) return;
        }
        copy(line_coding, ep0_out_buf, 7);
        EPINTFLAG(0) = EPINT_TRCPT0;
        touch_armed = (line_coding[0] == 0xB0 && line_coding[1] == 0x04 &&
                       line_coding[2] == 0x00 && line_coding[3] == 0x00);
        ep0_status_in();
        return;
      case 0x21:                                   /* GET_LINE_CODING */
        ep0_write(line_coding, 7, wLength);
        return;
      case 0x22:                                   /* SET_CONTROL_LINE_STATE */
        /* bit 0 is DTR. Armed at 1200 baud, fired when the host drops it —
           which is what closing the port does. */
        if (touch_armed && (wValue & 1u) == 0) {
          ep0_status_in();
          usb_cdc_reboot_bootloader();
        }
        ep0_status_in();
        return;
      case 0x23:                                   /* SEND_BREAK */
        ep0_status_in();
        return;
      default:
        ep0_stall();
        return;
    }
  }

  ep0_stall();
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
/* PUBLIC                                                                    */
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void usb_cdc_init(void) {
  uint32_t cal;

  USB_CTRLA = 1u;                                  /* SWRST */
  while (USB_SYNCBUSY & 1u) { }

  /* PA24 and PA25 are D- and D+, on peripheral function H. */
  REG8(0x41008000u + 0x40 + 24) = (1u << 0);       /* PINCFG24: PMUXEN */
  REG8(0x41008000u + 0x40 + 25) = (1u << 0);       /* PINCFG25: PMUXEN */
  REG8(0x41008000u + 0x30 + 12) = (7u << 0) | (7u << 4);

  /* Pad calibration comes from the NVM software calibration area. The
     datasheet's "if it reads all ones, use this instead" defaults are not
     paranoia: an uncalibrated pad is an eye diagram the host may or may not
     like, which is the kind of bug that reproduces on one machine. */
  cal = REG32(0x00800080u + 4);
  {
    uint32_t transn = cal & 0x1Fu;
    uint32_t transp = (cal >> 5) & 0x1Fu;
    uint32_t trim   = (cal >> 10) & 0x7u;
    if (transn == 0x1Fu) transn = 5;
    if (transp == 0x1Fu) transp = 29;
    if (trim   == 0x7u)  trim   = 3;
    USB_PADCAL = (uint16_t)(transp | (transn << 6) | (trim << 12));
  }

  USB_QOSCTRL = (3u << 0) | (3u << 2);             /* highest QoS both ways */
  USB_DESCADD = (uint32_t)(uintptr_t)ep_desc;

  USB_CTRLA = (1u << 1);                           /* ENABLE, device mode */
  while (USB_SYNCBUSY & (1u << 1)) { }

  configured = 0;
  ep_configure();

  /* Address zero, responding: the host resets before it asks anything, and
     EORST puts this back, but the device has to be answering before the first
     reset arrives rather than after it. */
  USB_DADD = 0x80u;

  /* Attach last. Clearing DETACH is what puts the pull-up on D+ and tells the
     host there is something here; doing it before the endpoints exist invites
     a setup packet the device is not yet able to answer. */
  USB_CTRLB = 0u;                                  /* full speed, attached */
}

void usb_cdc_task(void) {
  if (USB_INTFLAG & INTFLAG_EORST) {
    USB_INTFLAG = INTFLAG_EORST;
    configured = 0;
    touch_armed = 0;
    USB_DADD = 0x80u;                              /* address 0, enabled */
    ep_configure();
    return;
  }

  if (EPINTFLAG(0) & EPINT_RXSTP) handle_setup();

  /* Anything the host writes to the console is discarded, but the bank has to
     be given back or the endpoint stops accepting and the host's write blocks
     forever in its driver rather than failing. */
  if (EPINTFLAG(EP_DATA) & EPINT_TRCPT0) {
    EPINTFLAG(EP_DATA) = EPINT_TRCPT0;
    EPSTATUSCLR(EP_DATA) = EPST_BK0RDY;
  }

  if (tx_busy && (EPINTFLAG(EP_DATA) & EPINT_TRCPT1)) {
    EPINTFLAG(EP_DATA) = EPINT_TRCPT1;
    tx_busy = 0;
  }
}

int usb_cdc_configured(void) { return configured; }

/* Hand the buffered packet to the endpoint and wait for it to leave.
 *
 * Waiting rather than dropping is the deliberate choice. A record is a fixed
 * sequence of lines that the host parses and checks a nonce on; half a record
 * with the middle missing would still parse, and would be wrong in a way
 * nothing downstream could see. Blocking until the host reads means a board
 * nobody is listening to simply pauses, which the BEGIN/END framing already
 * handles — the harness waits for two complete records and reads the second.
 *
 * USB is serviced while blocked, so control traffic still gets answered; that
 * matters because this is where the firmware spends its time before anyone
 * has opened the port.
 */
static void tx_flush_blocking(void) {
  if (tx_len == 0) return;

  while (tx_busy) {
    usb_cdc_task();
    if (!configured) { tx_len = 0; tx_busy = 0; return; }
  }

  ep_desc[EP_DATA].bank[1].addr = (uint32_t)(uintptr_t)data_in_buf;
  ep_desc[EP_DATA].bank[1].pcksize = PCKSIZE_64 | tx_len;
  EPINTFLAG(EP_DATA) = EPINT_TRCPT1;
  EPSTATUSSET(EP_DATA) = EPST_BK1RDY;
  tx_busy = 1;
  tx_len = 0;
}

void usb_cdc_putc(char c) {
  if (!configured) {
    /* Not enumerated yet. Service the bus rather than filling a buffer that
       has nowhere to go; `board_init` does not return until this clears. */
    usb_cdc_task();
    return;
  }
  data_in_buf[tx_len++] = (uint8_t)c;
  if (tx_len >= MAX_PACKET) tx_flush_blocking();
}

void usb_cdc_flush(void) {
  tx_flush_blocking();
  /* Drain it too, so a caller that flushes per line leaves nothing in flight
     for the next line to wait behind. */
  while (tx_busy) {
    usb_cdc_task();
    if (!configured) return;
  }
}
