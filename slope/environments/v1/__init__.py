from slope.core import Environment
from slope.environments.v1.operators import operators_set
from slope.environments.v1.procedures import procedures_set
from slope.environments.v1.backends import numpy_backend

v1_environment = Environment(operators_set, procedures_set, dict(numpy=numpy_backend))
