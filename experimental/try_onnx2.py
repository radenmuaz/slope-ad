# A set of code samples showing different usage of the ONNX Runtime Python API
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnxruntime
import onnx
# from onnx import parser, printer

MODEL_FILE = '.model.onnx'
DEVICE_NAME = 'cpu'
DEVICE_INDEX = 0
DEVICE=f'{DEVICE_NAME}:{DEVICE_INDEX}'
SIZE = (1, 3, 32, 32)

model_text = """
<
   ir_version: 9,
   opset_import: ["" : 15, "slope" : 1]
>
agraph (int64[] shape) => (float[] y) {
   one = Constant <value = float[1] {1}> ()
   y = slope.full (one, shape)
}
<
  domain: "slope",
  opset_import: ["" : 1]
>
full (x, shape) => (y)
{
   y = Expand (x, shape)
}
"""
model = onnx.parser.parse_model(model_text)
model_text = onnx.printer.to_text(model)
print(model_text)

path = "/tmp/model.onnx"
onnx.save_model(model, path)
sess= onnxruntime.InferenceSession(path,  providers=['CPUExecutionProvider'])
input = dict( shape=np.array([2]) )
output_names = ['y']
out = sess.run(output_names, input)
for o in out:
    print(f"{o.shape=}\n{o=}")

# onnx.checker.check_model(model)
