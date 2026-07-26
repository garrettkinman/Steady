# Test fixture: a deliberately partial backend.
#
# It overrides fullyConnected for AffineI8 ONLY. Everything else — other ops,
# and fullyConnected under the real-number policies — must fall through to
# the reference kernels.
import steady/policy

var beCalls*: int = 0

proc fullyConnected*(_: typedesc[AffineI8],
                     y: ptr UncheckedArray[int8],
                     x: ptr UncheckedArray[int8],
                     w: ptr UncheckedArray[int8],
                     bias: ptr UncheckedArray[int32],
                     prm: AffineParams,
                     outDim, inDim: int) =
  inc beCalls
  for o in 0 ..< outDim:
    var acc = 0'i32
    for i in 0 ..< inDim:
      acc += int32(w[o * inDim + i]) * int32(x[i])
    acc += bias[o]
    y[o] = AffineI8.finish(acc, prm, o)
