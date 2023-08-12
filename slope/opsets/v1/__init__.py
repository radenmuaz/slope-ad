from slope.core import Opset
from slope.opsets.v1.ops_defs import ops
from slope.opsets.v1.procs_defs import procs
from slope.opsets.v1.backend_defs import numpy_backend

v1_opset = Opset(ops, procs, dict(numpy=numpy_backend))
