from slope.core import Environment
from slope.environments.v1.operators import operator_set
from slope.environments.v1.procedures import procedure_set
from slope.environments.v1.backends import numpy_backend

v1_environment = Environment(operator_set, procedure_set, dict(numpy=numpy_backend))
