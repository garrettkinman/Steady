# Copyright (c) 2026 Garrett Kinman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""Builds the int8 fixtures that cannot simply be downloaded.

`fetch.sh` handles models that exist as canonical published files. These two do
not:

  mobilenet_v2   Every public "quantized" MobileNetV2 is the TF1-era uint8
                 file, whose asymmetric weights this compiler refuses on
                 purpose. A genuine int8 conversion has to be made.
  fomo           Edge Impulse's FOMO exports are per-project artefacts, so
                 there is no canonical file to pin. The architecture is
                 published and small, so it is rebuilt here.

What is being tested is the *compiler*, not the model's accuracy, so the
calibration set is deterministic pseudo-random data rather than real images.
That gives realistic quantization parameters — a spread of per-channel scales,
non-trivial zero points, saturating activations — which is all a bit-exactness
or throughput test can use. FOMO's weights are untrained for the same reason:
its architecture is what exercises the compiler.

Needs TensorFlow, which is heavier than the interpreter the harness normally
uses:

    .venv/bin/pip install tensorflow-cpu
    .venv/bin/python tests/models/convert.py

Both files land in tests/models/ and are gitignored. The harness skips whatever
is absent, so not running this is a supported state.
"""

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260726


def representative_dataset(shape, count=64):
    """Deterministic calibration data.

    Uniform noise over the input range gives wide activation ranges and so
    per-channel scales that differ from one another, which is what the
    per-channel requantization path needs in order to be tested at all. Real
    images would give a better model and a weaker test.
    """
    rng = np.random.default_rng(SEED)

    def gen():
        for _ in range(count):
            x = rng.uniform(-1.0, 1.0, size=(1,) + tuple(shape)).astype(np.float32)
            yield [x]

    return gen


def to_int8_tflite(model, input_shape):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset(input_shape)
    # Full integer: every operator quantized, and the graph's own I/O in int8
    # so there is no boundary QUANTIZE to strip. `vww` and `resnet8` already
    # cover the float-boundary case, so these cover the other one.
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def build_mobilenet_v2():
    """MobileNetV2 1.0 at 224, ImageNet weights.

    The flagship: 3.4 MB of weights, a 400 KB single activation, ten residual
    adds, and a 1000-class softmax — which is the first thing here to make the
    exp table's fractional-bit budget do any work.
    """
    return tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), alpha=1.0, weights="imagenet"
    ), (224, 224, 3)


def build_fomo(num_classes=2, alpha=0.35, size=96):
    """FOMO: a truncated MobileNetV1 backbone with a per-cell classifier head.

    Edge Impulse's design. The backbone is cut where the stride reaches 8, and
    a 1x1 convolution turns each cell of the remaining grid into class scores —
    so the output is a `size/8` square grid of softmaxes rather than one
    vector, and there is no global pooling and no dense layer anywhere.

    That output shape is the interesting part for a compiler: a softmax over
    the channel axis of a 4-D tensor, evaluated once per cell.
    """
    backbone = tf.keras.applications.MobileNet(
        input_shape=(size, size, 3), alpha=alpha, weights=None, include_top=False
    )
    # conv_pw_5_relu is the last layer at stride 8 in a MobileNetV1; beyond it
    # the spatial grid halves again, which is coarser than FOMO wants.
    cut = backbone.get_layer("conv_pw_5_relu").output
    head = tf.keras.layers.Conv2D(
        num_classes + 1, kernel_size=1, activation="softmax", name="fomo_head"
    )(cut)
    model = tf.keras.Model(backbone.input, head, name="fomo")
    return model, (size, size, 3)


TARGETS = {
    "mobilenet_v2": build_mobilenet_v2,
    "fomo": build_fomo,
}


def main():
    wanted = sys.argv[1:] or list(TARGETS)
    for name in wanted:
        if name not in TARGETS:
            raise SystemExit(f"unknown target '{name}'; have {', '.join(TARGETS)}")
        dest = os.path.join(HERE, f"{name}.tflite")
        print(f"==> building {name}")
        model, shape = TARGETS[name]()
        blob = to_int8_tflite(model, shape)
        with open(dest, "wb") as f:
            f.write(blob)
        print(f"    {dest}  {len(blob) / 1024:.0f} KiB")

        interp = tf.lite.Interpreter(model_content=blob)
        interp.allocate_tensors()
        i, o = interp.get_input_details()[0], interp.get_output_details()[0]
        print(f"    input  {list(i['shape'])} {i['dtype'].__name__}")
        print(f"    output {list(o['shape'])} {o['dtype'].__name__}")


if __name__ == "__main__":
    main()
