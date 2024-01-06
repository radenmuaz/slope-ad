import os
from slope import core
import importlib

SLOPE_BACKEND = os.environ.get("SLOPE_BACKEND", "iree")
core.set_backend(importlib.import_module(f"slope.backends.{SLOPE_BACKEND}").backend)

from slope import nn

def __getattr__(attr):
    if attr in (globals_dict := globals()):
        core.dblog(f"Looking slope.{attr} in globals()", enable=core.backend.LOG_BACKEND)
        return globals_dict[attr]
    elif attr in vars(core):
        core.dblog(f"Looking slope.{attr} in core", enable=core.backend.LOG_BACKEND)
        return getattr(core, attr)
    elif attr in vars(core.backend.operator_set):
        core.dblog(f"Looking slope.{attr} in core.backend.operator_set", enable=core.backend.LOG_BACKEND)
        return getattr(core.backend.operator_set, attr)
    elif attr in vars(core.backend.procedure_set):
        core.dblog(f"Looking slope.{attr} in core.backend.procedure_set", enable=core.backend.LOG_BACKEND)
        return getattr(core.backend.procedure_set, attr)
    elif attr in vars(core.Backend):
        core.dblog(f"Looking slope.{attr} in core.backend", enable=core.backend.LOG_BACKEND)
        return getattr(core.backend, attr)
    elif attr in core.Tensor.dtype_names.keys():
        return core.Tensor.dtype_names[attr]
    elif attr in core.Tensor.das_mlir_shape_names.keys():
        return core.Tensor.das_mlir_shape_names[attr]

    raise NameError(attr)
