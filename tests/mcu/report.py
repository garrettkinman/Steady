# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Joins a board's cycle counts to the operator names the compiler already
# knows, and prints the result in the same shape as `nimble bench`.
#
# The device deliberately sends indices rather than names: an op name from a
# real TFLite model runs to several hundred characters, and sending 31 of them
# over a 115200 baud line to say something the host already knows would be
# slower than the benchmark.
#
#   report.py <generated model.nim> <serial capture> [expected nonce]

import re
import sys


def op_metadata(path):
    """Op names, kinds and MAC counts from the generated module's profile table."""
    ops = []
    pat = re.compile(
        r'OpProfile\(name: "(.*?)", kind: "(.*?)", macs: (\d+), outElems: (\d+)\)')
    for line in open(path, encoding="utf-8"):
        m = pat.search(line)
        if m:
            ops.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    return ops


def records(path):
    """Split the capture into complete BEGIN..END blocks, keeping the last."""
    blocks, cur = [], None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("BEGIN"):
            cur = [line]
        elif cur is not None:
            cur.append(line)
            if line == "END":
                blocks.append(cur)
                cur = None
    return blocks


def short(name, width=32):
    s = name.split(";")[0]
    if s.startswith("model/"):
        s = s[6:]
    return s if len(s) <= width else "…" + s[-(width - 1):]


def main():
    meta = op_metadata(sys.argv[1])
    blocks = records(sys.argv[2])
    want_nonce = sys.argv[3] if len(sys.argv) > 3 else None
    if not blocks:
        sys.exit("no complete BEGIN..END record in the capture")

    if want_nonce:
        # A record from the previously flashed image parses perfectly and is
        # completely wrong. Refuse anything that is not from this build.
        blocks = [b for b in blocks
                  if any(l.split()[1:2] == [want_nonce]
                         for l in b if l.startswith("nonce "))]
        if not blocks:
            sys.exit(f"no record carrying this build's nonce {want_nonce}; "
                     "the capture is from an older image")
    block = blocks[-1]

    header = {}
    configs = []
    cur = None
    for line in block:
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key in ("sysclk", "arena", "macs", "ops", "overhead"):
            header[key] = int(parts[1])
        elif key == "config":
            cur = {"label": " ".join(parts[1:]), "ops": {}}
            configs.append(cur)
        elif key == "checksum" and cur is not None:
            cur["checksum"] = parts[1]
        elif key == "whole" and cur is not None:
            cur["whole"] = int(parts[1])
        elif key == "op" and cur is not None:
            cur["ops"][int(parts[1])] = int(parts[2])

    hz = header.get("sysclk", 80_000_000)
    macs = header.get("macs", 0)
    print(f"    {header.get('ops', 0)} ops   {macs / 1e6:.2f} MMAC   "
          f"arena {header.get('arena', 0)} bytes   "
          f"{hz / 1e6:.0f} MHz   measurement overhead "
          f"{header.get('overhead', 0)} cycles")

    # Whole-model, one row per flash-accelerator configuration. The last row is
    # the cacheless number, which is the one the target class actually has.
    print()
    print(f"      {'flash config':<34} {'cycles':>12} {'ms':>8} "
          f"{'cyc/MAC':>8} {'MMAC/s':>8}")
    for c in configs:
        w = c.get("whole", 0)
        ms = 1e3 * w / hz
        cpm = w / macs if macs else 0
        print(f"      {c['label']:<34} {w:>12} {ms:>8.2f} {cpm:>8.2f} "
              f"{macs / (w / hz) / 1e6 if w else 0:>8.1f}")

    sums = {c["checksum"] for c in configs if "checksum" in c}
    if len(sums) == 1:
        print(f"      output checksum {sums.pop()} — identical in every configuration")
    else:
        print(f"      FAIL: output checksum differs between configurations: {sums}")
        sys.exit(1)

    # Per operator, from the cacheless configuration: the honest one.
    base = configs[-1]
    total = sum(base["ops"].values()) or 1
    print()
    print(f"      per operator, {base['label']}")
    print(f"      {'op':>3}  {'kind':<18} {'MMAC':>8} {'cycles':>10} "
          f"{'cyc/MAC':>8} {'%':>6}  layer")
    order = sorted(base["ops"], key=lambda i: -base["ops"][i])
    for i in order:
        cyc = base["ops"][i]
        share = 100.0 * cyc / total
        if i < len(meta):
            name, kind, m, _ = meta[i]
        else:
            name, kind, m = "?", "?", 0
        if share < 0.05 and order.index(i) > 12:
            continue
        cpm = f"{cyc / m:>8.2f}" if m else f"{'-':>8}"
        print(f"      {i:>3}  {kind[2:]:<18} {m / 1e6:>8.3f} {cyc:>10} "
              f"{cpm} {share:>6.2f}  {short(name)}")

    # Per kind, the rollup that actually directs the work.
    by_kind = {}
    for i, cyc in base["ops"].items():
        if i >= len(meta):
            continue
        _, kind, m, _ = meta[i]
        k = by_kind.setdefault(kind[2:], [0, 0, 0])
        k[0] += 1
        k[1] += m
        k[2] += cyc
    print()
    print(f"      {'by kind':<20} {'n':>3} {'MMAC':>9} {'cycles':>11} "
          f"{'cyc/MAC':>8} {'%':>6}")
    for kind, (n, m, cyc) in sorted(by_kind.items(), key=lambda kv: -kv[1][2]):
        cpm = f"{cyc / m:>8.2f}" if m else f"{'-':>8}"
        print(f"      {kind:<20} {n:>3} {m / 1e6:>9.3f} {cyc:>11} {cpm} "
              f"{100.0 * cyc / total:>6.2f}")


if __name__ == "__main__":
    main()
