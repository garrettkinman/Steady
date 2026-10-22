# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

## Sizes in terms of a length given at run time.

import std/unittest
import steadyc/symbolic

let n = lengthDim()

suite "Dim":
  test "a fixed size is a constant, and prints as its number":
    check dim(64).isConst
    check dim(64).value == 64
    check $dim(64) == "64"
    check $(dim(3) * dim(4)) == "12"

  test "arithmetic in terms of the length":
    check $n == "(n)"
    check $(n * (n + 9)) == "(n*n + 9*n)"
    check $(2 * n * n - n) == "(2*n*n - n)"
    check (n * (n + 9)).at(64) == 64 * 73

  test "a reshape's -1 divides the whole by the rest":
    check (n * (n + 9) * 2) div (n * 2) == n + 9
    check (dim(2) * n * 16) div dim(16) == 2 * n
    expect SymbolicError:
      discard (n * n + 1) div n

  test "equality ignores how a size was made":
    check n * 2 - n == n
    check n - n == dim(0)
    check not n.isConst
