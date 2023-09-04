from slope.core import Env
from slope.envs.v1.ops_defs import ops
from slope.envs.v1.procs_defs import procs
from slope.envs.v1.backend_defs import numpy_backend

v1_env = Env(ops, procs, dict(numpy=numpy_backend))
