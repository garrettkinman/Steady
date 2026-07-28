# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Posit(8,0) arithmetic, and the exactness claim the policy rests on.
##
## The format tests are exhaustive rather than sampled: 256 encodings is few
## enough that there is no reason to check a subset and then wonder about the
## rest. The quire tests are the interesting ones — they check that a
## reduction is *exact*, which is the property that distinguishes this policy
## from `RealFp8` and the only reason to prefer a posit at eight bits.

import std/[unittest, math, algorithm]
import steady

suite "posit(8,0) encoding":

  test "the encodings everyone knows":
    check Posit8(0x00'u8).toFloat64 == 0.0
    check Posit8(0x40'u8).toFloat64 == 1.0
    check Posit8(0xC0'u8).toFloat64 == -1.0
    check Posit8(0x20'u8).toFloat64 == 0.5
    check Posit8(0x60'u8).toFloat64 == 2.0
    check Posit8(0x7F'u8).toFloat64 == 64.0          # maxpos = useed^(n-2)
    check Posit8(0x81'u8).toFloat64 == -64.0
    check Posit8(0x01'u8).toFloat64 == 1.0 / 64.0    # minpos
    check Posit8(0x7E'u8).toFloat64 == 32.0
    check Posit8(0x80'u8).isNaR

  test "every value is an exact multiple of 2^-6":
    # The claim the whole policy is built on: it is what makes a product a
    # multiple of 2^-12 and therefore what makes the quire exact.
    for b in 0 .. 255:
      let p = Posit8(uint8(b))
      if p.isNaR: continue
      check abs(units(p)) <= 4096
      check float64(units(p)) / 64.0 == p.toFloat64

  test "two's complement ordering is value ordering":
    # Free for posits, and the reason `clamp1d` and `maxPool2d` need no fold
    # over this type where fp8 needs one.
    var prev = -1.0e30
    for i in -128 .. 127:
      let p = Posit8(cast[uint8](int8(i)))
      if p.isNaR: continue
      check p.toFloat64 > prev
      prev = p.toFloat64

  test "sorting by encoding sorts by value":
    var xs: seq[Posit8]
    for b in 0 .. 255:
      let p = Posit8(uint8(b))
      if not p.isNaR: xs.add p
    xs.sort(proc (a, b: Posit8): int = (if a < b: -1 elif b < a: 1 else: 0))
    for i in 1 ..< xs.len:
      check xs[i - 1].toFloat64 < xs[i].toFloat64

suite "posit(8,0) rounding":

  test "round-trips every finite encoding":
    for b in 0 .. 255:
      let p = Posit8(uint8(b))
      if p.isNaR: continue
      check toPosit8(p.toFloat64).bits == p.bits
      check positFromQuire(Quire(units(p)) shl UnitScale).bits == p.bits

  test "saturates instead of overflowing — there is no infinity":
    check toPosit8(1.0e9).bits == Posit8Max.bits
    check toPosit8(-1.0e9).bits == Posit8Min.bits
    check toPosit8(64.5).bits == Posit8Max.bits
    check positFromQuire(high(Quire)).bits == Posit8Max.bits
    check positFromQuire(low(Quire)).bits == Posit8Min.bits

  test "a nonzero never rounds to zero":
    # No subnormals and no gradual underflow, so flushing would turn a small
    # weight into an absent one.
    check toPosit8(1.0e-9).bits == Posit8MinPos.bits
    check toPosit8(-1.0e-9).bits == 0xFF'u8
    check positFromQuire(Quire(1)).bits == Posit8MinPos.bits
    check positFromQuire(Quire(0)).bits == Posit8Zero.bits

  test "ties go to the even *encoding*, not the even fraction":
    # Where the fraction field is non-empty the two rules coincide.
    check toPosit8(1.015625).bits == 0x40'u8      # 1 .. 1.03125 -> 1
    check toPosit8(1.046875).bits == 0x42'u8      # 1.03125 .. 1.0625 -> 1.0625
    # Where it is empty they do not: the low bit is the last regime bit, and
    # at the bottom of the range that inverts the answer.
    check toPosit8(48.0).bits == 0x7E'u8          # 32 .. 64 -> 32  (0x7E even)
    check toPosit8(1.5 / 64.0).bits == 0x02'u8    # minpos .. 2*minpos -> up
    check positFromQuire(Quire(48 * QuireOne)).bits == 0x7E'u8
    check positFromQuire(Quire(96)).bits == 0x02'u8

  test "rounds to the nearest representable value, everywhere":
    proc nearest(x: float64): Posit8 =
      var best = Posit8MinPos
      var bestd = 1.0e30
      for b in 0 .. 255:
        let p = Posit8(uint8(b))
        if p.isNaR or p.bits == 0: continue      # zero is not a rounding target
        let d = abs(p.toFloat64 - x)
        if d < bestd:
          bestd = d
          best = p
      best
    for n in 64 .. 262144:                        # minpos .. maxpos, on the grid
      if n mod 7 != 0: continue                   # a stride, for the runtime
      let x = float64(n) / float64(QuireOne)
      let got = toPosit8(x)
      # Ties are pinned above; here only the distance has to be right.
      check abs(got.toFloat64 - x) == abs(nearest(x).toFloat64 - x)

  test "the two encoders are the same function":
    # `toPosit8` runs on the host and encodes every weight; `positFromQuire`
    # runs on the target and rounds every activation. They are derived
    # independently — one by halving a float, one by scanning an integer — so
    # this is a real cross-check and not a tautology.
    for n in -262200 .. 262200:
      if n == 0: continue
      check positFromQuire(Quire(n)).bits ==
            toPosit8(float64(n) / float64(QuireOne)).bits

suite "the quire":

  test "a reduction is exact, not merely wide":
    # 0.1 is not representable, so accumulating in the storage type compounds
    # the error and accumulating in float32 rounds 64 times. The quire rounds
    # once, at the end, so the answer is the correctly rounded exact sum.
    const N = 64
    var x, w: array[N, Posit8]
    for i in 0 ..< N:
      x[i] = toPosit8(0.1)
      w[i] = toPosit8(1.0)
    var acc = zeroAccum(RealP8)
    for i in 0 ..< N:
      mac(RealP8, acc, x[i], w[i])
    let exact = float64(N) * toPosit8(0.1).toFloat64
    check acc == Quire(round(exact * float64(QuireOne)))
    check finish(RealP8, acc, RealP8.noClamp, 0).bits == toPosit8(exact).bits

  test "mac is exact for every pair of encodings":
    for a in 0 .. 255:
      for b in 0 .. 255:
        let pa = Posit8(uint8(a))
        let pb = Posit8(uint8(b))
        var acc = zeroAccum(RealP8)
        mac(RealP8, acc, pa, pb)
        # NaR decodes as zero in arithmetic; the policy documents that.
        let fa = if pa.isNaR: 0.0 else: pa.toFloat64
        let fb = if pb.isNaR: 0.0 else: pb.toFloat64
        check acc == Quire(round(fa * fb * float64(QuireOne)))

  test "the headroom claim holds at full scale":
    # maxpos squared is 2^24 quire units, so an int64 takes 2^38 of them.
    const Product = Quire(64 * 64) * QuireOne
    check Product == Quire(1) shl 24
    check high(Quire) div Product >= Quire(1) shl 38

  test "addBias and accumulate are exact widenings":
    for b in 0 .. 255:
      let p = Posit8(uint8(b))
      let f = if p.isNaR: 0.0 else: p.toFloat64
      var acc = zeroAccum(RealP8)
      addBias(RealP8, acc, p)
      check acc == Quire(round(f * float64(QuireOne)))
      var acc2 = zeroAccum(RealP8)
      accumulate(RealP8, acc2, p)
      check acc2 == acc

  test "the clamp happens before the rounding, in the quire domain":
    # ReLU6 on a value above 6: the bound is exact on the 2^-12 grid, so the
    # result is exactly 6 rather than the rounding of something near it.
    let prm = RealParams[Quire](actMin: 0'i64, actMax: 6 * QuireOne)
    check finish(RealP8, Quire(9 * QuireOne), prm, 0).bits == toPosit8(6.0).bits
    check finish(RealP8, Quire(-3 * QuireOne), prm, 0).bits == Posit8Zero.bits

  test "division rounds on the quire grid, ties to even":
    # The one inexact step in the policy, and the only one that rounds twice.
    check divAccum(RealP8, Quire(7), 2) == Quire(4)      # 3.5 -> 4 (even)
    check divAccum(RealP8, Quire(5), 2) == Quire(2)      # 2.5 -> 2 (even)
    check divAccum(RealP8, Quire(-7), 2) == Quire(-4)    # symmetric
    check divAccum(RealP8, Quire(-5), 2) == Quire(-2)
    # A power-of-two count divides exactly: every quire value a sum of posits
    # can hold is a multiple of 2^6, so a 2x2 pool never rounds here at all.
    check divAccum(RealP8, Quire(64 * 5), 4) == Quire(80)

suite "posit policy against the kernels":

  template pa[T](a: var openArray[T]): ptr UncheckedArray[T] =
    cast[ptr UncheckedArray[T]](addr a[0])

  test "fullyConnected computes W*x + b":
    # Values chosen inside the accurate band around 1.0, where a posit8 has
    # five fraction bits and these results are exact.
    var y = [Posit8Zero, Posit8Zero]
    var x = [toPosit8(0.5), toPosit8(0.25)]
    var w = [toPosit8(1.5), toPosit8(0.5), toPosit8(1.0), toPosit8(1.0)]
    var b = [toPosit8(0.125), Posit8Zero]
    fullyConnected(RealP8, pa(y), pa(x), pa(w), pa(b), RealP8.noClamp, 2, 2)
    check y[0].toFloat64 == 1.0                  # 0.75 + 0.125 + 0.125
    check y[1].toFloat64 == 0.75

  test "tapered precision is real, and the far end of the range is coarse":
    # The cost of the dynamic range: above 16 the spacing is 8, so a sum of
    # 30 is stored as 32. This is the property that decides where a model's
    # activations have to sit, and it is worth pinning rather than
    # discovering.
    var y = [Posit8Zero]
    var x = [toPosit8(4.0), toPosit8(2.0)]
    var w = [toPosit8(6.0), toPosit8(3.0)]
    var b = [Posit8Zero]
    fullyConnected(RealP8, pa(y), pa(x), pa(w), pa(b), RealP8.noClamp, 1, 2)
    check y[0].toFloat64 == 32.0                 # the exact 30 rounds here
    check toPosit8(30.0).toFloat64 == 32.0
    check toPosit8(1.0 + 1.0 / 32.0).toFloat64 == 1.03125   # and near 1 it is fine

  test "relu clamps under posit ordering":
    var v = [toPosit8(-3.0), toPosit8(0.5), toPosit8(9.0)]
    clamp1d(RealP8, pa(v), pa(v), 3, toPosit8(0.0), toPosit8(6.0))
    check v[0].toFloat64 == 0.0
    check v[1].toFloat64 == 0.5
    check v[2].toFloat64 == 6.0

  test "maxPool2d ignores NaR rather than being swallowed by it":
    # NaR is 0x80, which sorts below -maxpos, so it loses every comparison.
    var y = [Posit8Zero]
    var x = [Posit8NaR, toPosit8(-2.0), toPosit8(0.25), toPosit8(-8.0)]
    maxPool2d(RealP8, pa(y), pa(x), 2, 2, 1, 1, 1, 2, 2, 1, 1, 0, 0,
              Posit8Min, Posit8Max)
    check y[0].toFloat64 == 0.25

  test "a lookup table is keyed on the raw encoding":
    var table: array[256, Posit8]
    for i in 0 ..< 256:
      table[i] = toPosit8(Posit8(uint8(i)).toFloat64 * 2.0)
    table[int(Posit8NaR.bits)] = Posit8NaR
    var y = [Posit8Zero, Posit8Zero, Posit8Zero]
    var x = [toPosit8(1.0), toPosit8(-3.0), toPosit8(0.5)]
    lut1d(RealP8, pa(y), pa(x), 3, pa(table))
    check y[0].toFloat64 == 2.0
    check y[1].toFloat64 == -6.0
    check y[2].toFloat64 == 1.0

  test "one kernel source, two numeric behaviours":
    # `fullyConnected` is written once and instantiated per policy. float32
    # has the value exactly; posit8 does not have it at all.
    var yf = [0'f32, 0'f32]
    var xf = [4'f32, 2'f32]
    var wf = [6'f32, 3'f32, 1'f32, 1'f32]
    var bf = [0'f32, 0'f32]
    fullyConnected(RealF32, pa(yf), pa(xf), pa(wf), pa(bf), RealF32.noClamp, 2, 2)
    check yf[0] == 30'f32

    var y8 = [Posit8Zero, Posit8Zero]
    var x8 = [toPosit8(4.0), toPosit8(2.0)]
    var w8 = [toPosit8(6.0), toPosit8(3.0), toPosit8(1.0), toPosit8(1.0)]
    var b8 = [Posit8Zero, Posit8Zero]
    fullyConnected(RealP8, pa(y8), pa(x8), pa(w8), pa(b8), RealP8.noClamp, 2, 2)
    check y8[0].toFloat64 == 32.0    # posit8 has no 30; float32 does
