# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Sizes that depend on a length given when the model runs.
##
## A model compiled with a bound takes its length, `n`, at run time, from 1
## to the bound. Every size in it is then a polynomial in `n` with integer
## coefficients: a sequence of `n` frames is `n`, attention's scores are
## `2 n^2`, and the relative-position trick that skews them reshapes through
## `n (n + 9)`. A `Dim` is one such polynomial. A fixed size is a constant
## one, so a model with no length at all is the case where every `Dim` is a
## constant, and nothing downstream needs to tell the two apart.
##
## The emitter writes a `Dim` as a Nim expression in `n`; the arena planner
## sizes each buffer at the bound, which is where the planner's checks make
## sure each is largest.

import std/[strutils, sequtils]

type
  Dim* = object
    coeffs: seq[int]   ## Of n^0, n^1, ..., without trailing zeros

  SymbolicError* = object of CatchableError

func trimmed(c: seq[int]): Dim =
  var c = c
  while c.len > 0 and c[^1] == 0: c.setLen(c.len - 1)
  Dim(coeffs: c)

func dim*(k: int): Dim = trimmed(@[k])
  ## A fixed size

func lengthDim*(): Dim = Dim(coeffs: @[0, 1])
  ## The length itself, `n`

func isConst*(d: Dim): bool = d.coeffs.len <= 1

func value*(d: Dim): int =
  ## A fixed size's value
  if not d.isConst:
    raise newException(SymbolicError, "a size that depends on the length has no one value")
  if d.coeffs.len == 0: 0 else: d.coeffs[0]

func degree*(d: Dim): int = d.coeffs.len - 1

func `==`*(a, b: Dim): bool = a.coeffs == b.coeffs

func `+`*(a, b: Dim): Dim =
  var c = newSeq[int](max(a.coeffs.len, b.coeffs.len))
  for i, v in a.coeffs: c[i] += v
  for i, v in b.coeffs: c[i] += v
  trimmed(c)

func `-`*(a: Dim): Dim = trimmed(a.coeffs.mapIt(-it))
func `-`*(a, b: Dim): Dim = a + -b

func `*`*(a, b: Dim): Dim =
  if a.coeffs.len == 0 or b.coeffs.len == 0: return dim(0)
  var c = newSeq[int](a.coeffs.len + b.coeffs.len - 1)
  for i, x in a.coeffs:
    for j, y in b.coeffs: c[i + j] += x * y
  trimmed(c)

func `+`*(a: Dim, k: int): Dim = a + dim(k)
func `-`*(a: Dim, k: int): Dim = a - dim(k)
func `*`*(a: Dim, k: int): Dim = a * dim(k)
func `*`*(k: int, a: Dim): Dim = a * dim(k)

func `div`*(a, b: Dim): Dim =
  ## `a / b`, which has to come out whole: what a reshape's -1 or a row count
  ## divides is always a product that includes the divisor
  if b.coeffs.len == 0:
    raise newException(SymbolicError, "division of a size by zero")
  var r = a.coeffs
  var q = newSeq[int](max(0, r.len - b.coeffs.len + 1))
  for i in countdown(q.high, 0):
    let lead = r[i + b.coeffs.high]
    if lead mod b.coeffs[^1] != 0: break
    q[i] = lead div b.coeffs[^1]
    for j, y in b.coeffs: r[i + j] -= q[i] * y
  if trimmed(r).coeffs.len != 0:
    raise newException(SymbolicError, "a size does not divide another")
  trimmed(q)

func at*(d: Dim, n: int): int =
  ## The size for a length of `n`
  var p = 1
  for v in d.coeffs:
    result += v * p
    p *= n

func product*(ds: openArray[Dim]): Dim =
  result = dim(1)
  for d in ds: result = result * d

func `$`*(d: Dim): string =
  ## As a Nim expression in `n`, or the number when the size is fixed
  if d.coeffs.len == 0: return "0"
  if d.isConst: return $d.coeffs[0]
  var terms: seq[string]
  for i in countdown(d.coeffs.high, 0):
    let c = d.coeffs[i]
    if c == 0: continue
    let power = if i == 0: "" else: repeat("n", i).toSeq.mapIt($it).join("*")
    terms.add(
      if i == 0: $c
      elif c == 1: power
      elif c == -1: "-" & power
      else: $c & "*" & power)
  "(" & terms.join(" + ").replace("+ -", "- ") & ")"
