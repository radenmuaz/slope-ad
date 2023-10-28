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


full_text = """
<
domain: "slope",
opset_import: [ "" : 1]
>
full (x,shape) => (y) {
    y = Expand(x, shape)
}
"""
functions = [onnx.parser.parse_function(full_text)]

graph_text = '''
agraph (int64[] shape) => (float[] y) {    
    one = Constant <value=float[1]{1}>()
    y = slope.full(one, shape)
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
path = "/tmp/model.onnx"
onnx.save_model(model, path)
sess= onnxruntime.InferenceSession(path,  providers=['CPUExecutionProvider'])
input = dict( shape=np.array([2]) )
output_names = ['y']
out = sess.run(output_names, input)
for o in out:
    print(f"{o.shape=}\n{o=}")

# onnx.checker.check_model(model)




#  <
#     ir_version: 7,
#     opset_import: ["" : 14]
# >
# model (int64[] shape) => (float y) {    
#     one = Constant <value=float[1]{1}>()
#     y = slope.full(one, shape)
# }

#  <
#           domain: "slope",
#           opset_import: [ "" : 15],
#           doc_string: "Test function proto"
# >
# full (float[1] x, int64[] shape) => (float y) {
#     y = Expand(x, shape)
# }
# '''
# model = onnx.parser.parse_model(text)
# text = onnx.printer.to_text(model)
# print(text)

# model = onnx.parser.parse_model(code)

# text = onnx.printer.to_text(model)
# path = "/tmp/model.onnx"
# onnx.save_model(model, path)
# print(text)
# sess= onnxruntime.InferenceSession(path,  providers=['CPUExecutionProvider'])
# input = dict( shape=np.array([2,1]) )
# out = sess.run(['y'], input)
# for o in out:
#     print(f"{o.shape=}\n{o=}")
