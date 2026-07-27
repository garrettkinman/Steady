/* Copyright (c) 2026 Garrett Kinman
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 *
 * A C consumer of a packaged model, written the way a firmware project would
 * write it: include one header, link one archive, no Nim in the build and no
 * knowledge of what generated any of it.
 *
 * Everything here comes from the header — buffer sizes, shapes, the arena
 * size, the quantization. If the header is not self-sufficient this file does
 * not compile, which is the point of it.
 */

#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#include "tiny_cnn.h"

static const int input_shape[STEADY_TINY_CNN_INPUT0_RANK] =
    STEADY_TINY_CNN_INPUT0_SHAPE;

int main(void) {
  steady_tiny_cnn_init();

  /* The header's numbers must agree with the library's. */
  if (steady_tiny_cnn_arena_size() != (size_t)STEADY_TINY_CNN_ARENA_SIZE) {
    printf("FAIL: arena size %zu != %d\n", steady_tiny_cnn_arena_size(),
           STEADY_TINY_CNN_ARENA_SIZE);
    return 1;
  }

  int elems = 1;
  for (int d = 0; d < STEADY_TINY_CNN_INPUT0_RANK; d++) elems *= input_shape[d];
  if (elems != STEADY_TINY_CNN_INPUT0_ELEMS) {
    printf("FAIL: shape product %d != %d\n", elems, STEADY_TINY_CNN_INPUT0_ELEMS);
    return 1;
  }
  if (STEADY_TINY_CNN_INPUT0_BYTES != STEADY_TINY_CNN_INPUT0_ELEMS * (int)sizeof(int8_t)) {
    printf("FAIL: byte count disagrees with element count\n");
    return 1;
  }
  if (!(STEADY_TINY_CNN_INPUT0_SCALE > 0.0f)) {
    printf("FAIL: non-positive input scale\n");
    return 1;
  }

  int8_t *in = steady_tiny_cnn_input0();
  for (int i = 0; i < STEADY_TINY_CNN_INPUT0_ELEMS; i++) {
    in[i] = (int8_t)(i - 32);
  }

  steady_tiny_cnn_invoke();

  const int8_t *out = steady_tiny_cnn_output0();
  for (int i = 0; i < STEADY_TINY_CNN_OUTPUT0_ELEMS; i++) {
    printf("%d\n", (int)out[i]);
  }

  int best = 0;
  for (int i = 1; i < STEADY_TINY_CNN_OUTPUT0_ELEMS; i++) {
    if (out[i] > out[best]) best = i;
  }
  printf("argmax %d\n", best);

  /* Dequantize with the macros the header supplies — the reason they are
     there at all is that a caller holding real values needs them. */
  float logit = (float)(out[best] - STEADY_TINY_CNN_OUTPUT0_ZERO_POINT) *
                STEADY_TINY_CNN_OUTPUT0_SCALE;
  if (!(logit > -1e30f && logit < 1e30f)) {
    printf("FAIL: dequantized logit is not finite\n");
    return 1;
  }

  /* Repeatability: the model carries no state between invocations. Note the
     input has to be written again first — the header says input buffers are
     scratch, and the arena really does reuse them. */
  int8_t first[STEADY_TINY_CNN_OUTPUT0_ELEMS];
  for (int i = 0; i < STEADY_TINY_CNN_OUTPUT0_ELEMS; i++) first[i] = out[i];
  for (int i = 0; i < STEADY_TINY_CNN_INPUT0_ELEMS; i++) in[i] = (int8_t)(i - 32);
  steady_tiny_cnn_invoke();
  for (int i = 0; i < STEADY_TINY_CNN_OUTPUT0_ELEMS; i++) {
    if (out[i] != first[i]) {
      printf("FAIL: invoke is not repeatable at %d\n", i);
      return 1;
    }
  }

  return 0;
}
