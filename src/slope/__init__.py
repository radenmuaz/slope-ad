"""
.. include:: ../../README.md
"""
__all__ = ["core"]
__docformat__ = "markdown"
import os
from slope import core
import importlib
import numpy as np
np.set_printoptions(precision=5, threshold=1000, edgeitems=5, linewidth=120)
# SLOPE_BACKEND = os.environ.get("SLOPE_BACKEND", "iree")
SLOPE_BACKEND = os.environ.get("SLOPE_BACKEND", "onnxruntime")
# SLOPE_BACKEND = os.environ.get("SLOPE_BACKEND", "numpy")
# try:
core.set_backend(SLOPE_BACKEND)

# except Exception as e:
#     import traceback

#     traceback.print_stack()
#     print(e)
#     print(f"\n  -- Warning: failed to set {SLOPE_BACKEND} as backend. Error above \n")


def __getattr__(attr):
    if attr in (globals_dict := globals()):
        core.dblog(
            f"Looking slope.{attr} in globals()",
            enable=core.backend.LOG_BACKEND,
        )
        return globals_dict[attr]
    elif attr in vars(core):
        core.dblog(f"Looking slope.{attr} in core", enable=core.backend.LOG_BACKEND)
        return getattr(core, attr)
    elif attr in vars(core.backend.operator_set):
        core.dblog(
            f"Looking slope.{attr} in core.backend.operator_set",
            enable=core.backend.LOG_BACKEND,
        )
        return getattr(core.backend.operator_set, attr)
    elif attr in vars(core.backend.procedure_set):
        core.dblog(
            f"Looking slope.{attr} in core.backend.procedure_set",
            enable=core.backend.LOG_BACKEND,
        )
        return getattr(core.backend.procedure_set, attr)
    elif attr in vars(core.Backend):
        core.dblog(
            f"Looking slope.{attr} in core.backend",
            enable=core.backend.LOG_BACKEND,
        )
        return getattr(core.backend, attr)
    elif attr in core.dtypes.name_dtype_map.keys():
        return core.dtypes.name_dtype_map[attr]
    elif attr in core.dtypes.mlir_dtype_map.keys():
        return core.dtypes.mlir_dtype_map[attr]

    raise NameError(attr)
