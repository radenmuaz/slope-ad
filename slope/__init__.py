from slope import core
import os

LOG_LRU = int(os.environ.get("LOG_LRU", 0))
LOG_JIT = int(os.environ.get("LOG_JIT", 0))
LOG_PYTREE = int(os.environ.get("LOG_PYTREE", 0))
LOG_BACKEND = int(os.environ.get("LOG_BACKEND", 0))
LOG_INIT = int(os.environ.get("LOG_INIT", 0))
INLINE_PROCEDURE = int(os.environ.get("INLINE_PROCEDURE", 0))
SLOPE_DEVICE = os.environ.get("SLOPE_DEVICE", "cpu")
SLOPE_DTYPE = core.Tensor.dtype_names[os.environ.get("SLOPE_DTYPE", "float32")]
SLOPE_BACKEND = os.environ.get("SLOPE_BACKEND", "numpy")
NO_JIT = int(os.environ.get("NO_JIT", 0))

def dblog(*msg, enable=True):
    if enable:
        print(*msg)


class LazyInitMachine:
    def __getattr__(self, attr):
        return getattr(M(), attr)


machine = LazyInitMachine()


def M():
    global machine
    if type(machine) is LazyInitMachine:
        # import here to avoid circular import
        from slope.backends.numpy_backend import numpy_backend
        from slope.backends.onnxruntime_backend import onnxruntime_backend

        backend_registry = dict(numpy=numpy_backend, onnxruntime=onnxruntime_backend)
        if SLOPE_BACKEND not in backend_registry.keys():
            raise ValueError(f"{SLOPE_BACKEND} is nonexistent backend in: {list(backend_registry.keys())}")
        machine = core.Machine(backend=backend_registry[SLOPE_BACKEND])
        dblog(f"Auto init with {machine}", enable=LOG_INIT)
    return machine


def manual_init(init_machine):
    """
    Example usage:
    import slope
    from slope.backends.onnxruntime_backend import onnxruntime_backend
    slope.manual_init(slope.core.Machine(backend=onnxruntime_backend))
    """
    global machine
    machine = init_machine
    dblog(f"Manual init with {machine}", enable=LOG_INIT)


def __getattr__(attr):
    if attr in (globals_dict := globals()):
        return globals_dict[attr]
    M()

    if attr in vars(core.Machine):
        return getattr(machine, attr)

    if attr in vars(machine.backend.operator_set):
        return getattr(machine.backend.operator_set, attr)
    elif attr in vars(machine.backend.procedure_set):
        return getattr(machine.backend.procedure_set, attr)
    elif attr in vars(core.Backend):
        return getattr(machine.backend, attr)

    elif attr in core.Tensor.dtype_names.keys():
        return core.Tensor.dtype_names[attr]
    elif attr in core.Tensor.dtype_short_names.keys():
        return core.Tensor.dtype_short_names[attr]

    raise NameError(attr)
