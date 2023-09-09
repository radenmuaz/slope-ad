from slope.core import Environment
from slope.environments.v1.ops_defs import ops
from slope.environments.v1.procs_defs import procs
from slope.environments.v1.backend_defs import numpy_backend

v1_environment = Environment(ops, procs, dict(numpy=numpy_backend))
