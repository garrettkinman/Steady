# Test fixture: a deliberately partial *arithmetic* backend.
#
# The counterpart to steady_backend.nim, at the other seam. That one replaces
# whole kernels and sits above the policy; this one replaces policy members
# and sits below it, which is why it imports `steady/contract` rather than
# `steady/policy` — the module it is overriding imports *it*.
#
# It overrides two things for AffineI8:
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
# Everything else — every other member — must fall through to the defaults.
#
# This fixture matters more than its size suggests, and more than it did when
# it was written. It used to target one of four policies, and the abstraction
# had three other implementors standing as evidence that it was real. One
# policy ships now, so this is the *only* standing demonstration that the
# arithmetic seam is both substitutable and free: it replaces the arithmetic
# under the shipping policy, across the whole end-to-end suite, and demands
# the same bits out. Delete it and `mac` quietly stops being a primitive and
# becomes a spelling of `+=`.

import steady/contract

var arithCalls*: int = 0

template mac*(_: typedesc[AffineI8], acc: var int32, a, b: int8) =
  inc arithCalls
  acc = acc + int32(a) * int32(b)

template OcBlock*(_: typedesc[AffineI8]): int = 1
template DwBlock*(_: typedesc[AffineI8]): int = 1
template FcBlock*(_: typedesc[AffineI8]): int = 1
