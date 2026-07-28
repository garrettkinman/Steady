# Test fixture: a deliberately partial *arithmetic* backend.
#
# The counterpart to steady_backend.nim, at the other seam. That one replaces
# whole kernels and sits above the policy; this one replaces policy members
# and sits below it, which is why it imports `steady/contract` rather than
# `steady/policy` — the module it is overriding imports *it*.
#
# It overrides two things for RealP8 ONLY:
#
#   mac       with arithmetic identical to the default, so that every result
#             must stay bit-identical. What is under test is the seam, not a
#             new numeric behaviour.
#   OcBlock,  with 1, so the matmul-family kernels take their unblocked path.
#   DwBlock,  Results must *still* be bit-identical: each accumulator is
#   FcBlock   supposed to see the same taps in the same order whatever the
#             blocking, and this is what makes that claim testable rather
#             than asserted.
#
# Everything else — the other three policies, and every other member — must
# fall through to the defaults.

import steady/contract

var arithCalls*: int = 0

template mac*(_: typedesc[RealP8], acc: var Quire, a, b: Posit8) =
  inc arithCalls
  acc = acc + int64(units(a)) * int64(units(b))

template OcBlock*(_: typedesc[RealP8]): int = 1
template DwBlock*(_: typedesc[RealP8]): int = 1
template FcBlock*(_: typedesc[RealP8]): int = 1
