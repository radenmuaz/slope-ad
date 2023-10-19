from slope.core import Environment
from slope.environments.ov2.operators import operator_set
from slope.environments.ov2.procedures import procedure_set
from slope.environments.ov2.backends import onnxruntime_backend

v1_environment = Environment(operator_set, procedure_set, dict(onnxruntime=onnxruntime_backend))
