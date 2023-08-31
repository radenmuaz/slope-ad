from slope.core import System
from slope.systems.v1.ops_defs import ops
from slope.systems.v1.procs_defs import procs
from slope.systems.v1.backend_defs import numpy_backend

v1_system = System(ops, procs, dict(numpy=numpy_backend))
