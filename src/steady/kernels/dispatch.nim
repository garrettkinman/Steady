# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Backend dispatch.
##
## Generated code calls only through this module. Each entry point tries a
## vendor backend first and silently falls back to the portable reference
## kernel when the backend does not provide that op *for that policy*. The
## selection is a `when compiles(...)` test, so it costs nothing at runtime
## and produces no indirect calls — there is no registry and no vtable.
##
## This is the *kernel* seam. The one below it — replacing the arithmetic a
## kernel is built from, which is what a posit or fp8 arithmetic unit wants —
## is `kernels/arith.nim`. A backend can use either or both.
##
## To supply a backend:
##
##   1. write a module named `steady_backend` exposing any subset of the
##      entry points below, with matching signatures
##   2. build with `-d:steadyBackend --path:<dir containing it>`
##
## Overriding is per-op *and* per-policy: a backend that defines only an
## int8 `conv2d` will be used for int8 convolutions and ignored everywhere
## else, because the `compiles` test fails for the other instantiations and
## falls through. Nothing needs to be forked, and a backend never has to
## implement ops it does not care about.
##
## Note that a serious accelerator usually needs more than a faster kernel —
## CMSIS-NN wants particular weight orderings, an NPU wants weights
## pre-tiled, and both want them arranged at build time rather than on the
## device. That half of the story is a host-side hook in `steadyc`
## (see `steadyc/backend.nim`), not something this module can express.

import ./arith
from ./reference import nil

export arith

when defined(steadyBackend):
  from steady_backend import nil

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template dispatch(call, fallback: untyped): untyped =
  ## `compiles` is false for an undeclared `steady_backend`, so the guard
  ## works whether or not a backend was configured.
  when compiles(call):
    call
  else:
    fallback

template fullyConnected*(P: typedesc, y, x, w, bias, prm: untyped,
                         outDim, inDim: untyped) =
  dispatch(steady_backend.fullyConnected(P, y, x, w, bias, prm, outDim, inDim),
           reference.fullyConnected(P, y, x, w, bias, prm, outDim, inDim))

template conv2d*(P: typedesc, y, x, w, bias, prm: untyped,
                 inH, inW, inC, outH, outW, outC, kH, kW,
                 strideH, strideW, padTop, padLeft,
                 dilationH, dilationW, padValue: untyped) =
  dispatch(steady_backend.conv2d(P, y, x, w, bias, prm, inH, inW, inC,
                                 outH, outW, outC, kH, kW, strideH, strideW,
                                 padTop, padLeft, dilationH, dilationW, padValue),
           reference.conv2d(P, y, x, w, bias, prm, inH, inW, inC,
                            outH, outW, outC, kH, kW, strideH, strideW,
                            padTop, padLeft, dilationH, dilationW, padValue))

template depthwiseConv2d*(P: typedesc, y, x, w, bias, prm: untyped,
                          inH, inW, inC, outH, outW, kH, kW, depthMultiplier,
                          strideH, strideW, padTop, padLeft,
                          dilationH, dilationW, padValue: untyped) =
  dispatch(steady_backend.depthwiseConv2d(P, y, x, w, bias, prm, inH, inW, inC,
                                          outH, outW, kH, kW, depthMultiplier,
                                          strideH, strideW, padTop, padLeft,
                                          dilationH, dilationW, padValue),
           reference.depthwiseConv2d(P, y, x, w, bias, prm, inH, inW, inC,
                                     outH, outW, kH, kW, depthMultiplier,
                                     strideH, strideW, padTop, padLeft,
                                     dilationH, dilationW, padValue))

template maxPool2d*(P: typedesc, y, x: untyped,
                    inH, inW, channels, outH, outW, kH, kW,
                    strideH, strideW, padTop, padLeft, actMin, actMax: untyped) =
  dispatch(steady_backend.maxPool2d(P, y, x, inH, inW, channels, outH, outW,
                                    kH, kW, strideH, strideW, padTop, padLeft,
                                    actMin, actMax),
           reference.maxPool2d(P, y, x, inH, inW, channels, outH, outW,
                               kH, kW, strideH, strideW, padTop, padLeft,
                               actMin, actMax))

template avgPool2d*(P: typedesc, y, x: untyped,
                    inH, inW, channels, outH, outW, kH, kW,
                    strideH, strideW, padTop, padLeft, actMin, actMax: untyped) =
  dispatch(steady_backend.avgPool2d(P, y, x, inH, inW, channels, outH, outW,
                                    kH, kW, strideH, strideW, padTop, padLeft,
                                    actMin, actMax),
           reference.avgPool2d(P, y, x, inH, inW, channels, outH, outW,
                               kH, kW, strideH, strideW, padTop, padLeft,
                               actMin, actMax))

template clamp1d*(P: typedesc, y, x, n, lo, hi: untyped) =
  dispatch(steady_backend.clamp1d(P, y, x, n, lo, hi),
           reference.clamp1d(P, y, x, n, lo, hi))

template addElementwise*(P: typedesc, y, a, b, prm, n: untyped,
                         aMult, aShift, bMult, bShift, aZero, bZero: untyped) =
  dispatch(steady_backend.addElementwise(P, y, a, b, prm, n, aMult, aShift,
                                         bMult, bShift, aZero, bZero),
           reference.addElementwise(P, y, a, b, prm, n, aMult, aShift,
                                    bMult, bShift, aZero, bZero))

template copy1d*(P: typedesc, y, x, n: untyped) =
  dispatch(steady_backend.copy1d(P, y, x, n),
           reference.copy1d(P, y, x, n))

template pad2d*(P: typedesc, y, x: untyped,
                inH, inW, channels, padTop, padBottom, padLeft, padRight,
                padValue: untyped) =
  dispatch(steady_backend.pad2d(P, y, x, inH, inW, channels,
                                padTop, padBottom, padLeft, padRight, padValue),
           reference.pad2d(P, y, x, inH, inW, channels,
                           padTop, padBottom, padLeft, padRight, padValue))

template concatSlice*(P: typedesc, y, x: untyped,
                      outer, innerIn, innerOut, dstOffset: untyped) =
  dispatch(steady_backend.concatSlice(P, y, x, outer, innerIn, innerOut, dstOffset),
           reference.concatSlice(P, y, x, outer, innerIn, innerOut, dstOffset))

template concatSliceRescaled*(P: typedesc, y, x, prm: untyped,
                              outer, innerIn, innerOut, dstOffset,
                              mult, shift, offset: untyped) =
  dispatch(steady_backend.concatSliceRescaled(P, y, x, prm, outer, innerIn,
                                              innerOut, dstOffset,
                                              mult, shift, offset),
           reference.concatSliceRescaled(P, y, x, prm, outer, innerIn,
                                         innerOut, dstOffset,
                                         mult, shift, offset))

template meanSpatial*(P: typedesc, y, x, corr, prm: untyped,
                      inH, inW, channels: untyped) =
  dispatch(steady_backend.meanSpatial(P, y, x, corr, prm, inH, inW, channels),
           reference.meanSpatial(P, y, x, corr, prm, inH, inW, channels))

template lut1d*(P: typedesc, y, x, n, table: untyped) =
  dispatch(steady_backend.lut1d(P, y, x, n, table),
           reference.lut1d(P, y, x, n, table))

template softmax*(P: typedesc, y, x, rows, classes, expLut: untyped,
                  outMult, outShift, outZeroPoint, actMin, actMax: untyped) =
  dispatch(steady_backend.softmax(P, y, x, rows, classes, expLut, outMult,
                                  outShift, outZeroPoint, actMin, actMax),
           reference.softmax(P, y, x, rows, classes, expLut, outMult, outShift,
                             outZeroPoint, actMin, actMax))
