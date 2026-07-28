# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# Wrap a raw binary in the UF2 container a resident bootloader expects.
#
# The format is deliberately trivial — a stream of self-describing 512-byte
# blocks, each carrying 256 bytes of payload and its own destination address —
# because it has to survive being written by an operating system that thinks
# it is copying a file to a memory stick, in whatever order it likes. That is
# also why this is thirty lines here rather than a dependency: reimplementing
# it is less work than pinning someone else's copy of it.
#
#   uf2conv.py --base 0x4000 --family 0x55114460 -o image.uf2 image.bin

import argparse
import struct

MAGIC_START0 = 0x0A324655
MAGIC_START1 = 0x9E5D5157
MAGIC_END = 0x0AB16F30
FLAG_FAMILY_ID = 0x00002000

PAYLOAD = 256
BLOCK = 512


def convert(data, base, family):
    blocks = (len(data) + PAYLOAD - 1) // PAYLOAD
    out = bytearray()
    for i in range(blocks):
        chunk = data[i * PAYLOAD:(i + 1) * PAYLOAD]
        header = struct.pack(
            "<IIIIIIII",
            MAGIC_START0, MAGIC_START1, FLAG_FAMILY_ID,
            base + i * PAYLOAD, PAYLOAD, i, blocks, family)
        out += header
        out += chunk.ljust(PAYLOAD, b"\0")
        # Every block is the same size whether or not its payload fills it,
        # so the bootloader can seek by block number without parsing.
        out += b"\0" * (BLOCK - len(header) - PAYLOAD - 4)
        out += struct.pack("<I", MAGIC_END)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser(description="wrap a binary as UF2")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--base", required=True,
                    help="flash address of the first byte, e.g. 0x4000")
    ap.add_argument("--family", required=True,
                    help="UF2 family id the bootloader accepts")
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        data = f.read()
    uf2 = convert(data, int(args.base, 0), int(args.family, 0))
    with open(args.output, "wb") as f:
        f.write(uf2)
    print(f"    {len(data)} bytes -> {len(uf2) // BLOCK} UF2 blocks")


if __name__ == "__main__":
    main()
