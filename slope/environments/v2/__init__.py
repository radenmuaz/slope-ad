from slope.core import Environment
from slope.environments.v2.operators import operator_set
from slope.environments.v2.procedures import procedure_set
from slope.environments.v2.backends import numpy_backend

v1_environment = Environment(operator_set, procedure_set, dict(onnxruntime=onnxruntime_backend))
