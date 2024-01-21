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

# 1.

full_text = """
<
domain: "slope",
opset_import: [ "" : 1]
>
full (x,shape) => (y1, y2) {
    y1 = Expand(x, shape)
    y2 = Add(x, x)
}
"""
functions = [onnx.parser.parse_function(full_text)]

graph_text = '''
agraph (int64[] shape) => (float[] y1, float[] y2) {    
    one = Constant <value=float[1]{1}>()
    y1, y2 = slope.full(one, shape)
}
'''
graph = onnx.parser.parse_graph(graph_text)
opset_imports = [
            onnx.OperatorSetIdProto(domain="", version=15),
            onnx.OperatorSetIdProto(domain="slope", version=1),
        ]

model = onnx.helper.make_model(
    graph, functions=functions, opset_imports=opset_imports
)
model_text = onnx.printer.to_text(model)
print(model_text)

path = "/tmp/model.onnx"
onnx.save_model(model, path)
sess= onnxruntime.InferenceSession(path,  providers=['CPUExecutionProvider'])
input = dict( shape=np.array([2]) )
output_names = ['y1', 'y2']
out = sess.run(output_names, input)
for n, o in zip(output_names, out):
    print(f"{n}\nshape: {o.shape=}\ncontents: {o}\n")

# onnx.checker.check_model(model)
