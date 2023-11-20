from slope import core
import os

LOG_LRU = int(os.environ.get("LOG_LRU", 0))
LOG_JIT = int(os.environ.get("LOG_JIT", 0))
LOG_PYTREE = int(os.environ.get("LOG_PYTREE", 0))
LOG_ENV = int(os.environ.get("LOG_ENV", 0))
LOG_INIT = int(os.environ.get("LOG_INIT", 0))
INLINE_PROCEDURE = int(os.environ.get("INLINE_PROCEDURE", 0))
DEFAULT_DEVICE = os.environ.get("DEFAULT_DEVICE", "cpu")
DEFAULT_ENV = os.environ.get("DEFAULT_ENV", "numpy")

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
        from slope.environments.numpy_environment import numpy_environment
        from slope.environments.onnxruntime_environment import onnxruntime_environment
        environment_registry = dict(
            numpy=numpy_environment,
            onnxruntime=onnxruntime_environment
        )
        if DEFAULT_ENV not in environment_registry.keys():
            raise ValueError(f"{DEFAULT_ENV} isnonexistent environment in: {list(environment_registry.keys())}")
        machine = core.Machine(environment=environment_registry[DEFAULT_ENV])
        dblog(f"Auto init with {machine}", enable=LOG_INIT)
    return machine


def manual_init(init_machine):
    """
    Example usage:
    from slope.environments.onnxruntime_environment import onnxruntime_environment
    slope.manual_init(slope.core.Machine(environment=onnxruntime_environment))
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

    if attr in vars(machine.environment.operator_set):
        return getattr(machine.environment.operator_set, attr)
    elif attr in vars(machine.environment.procedure_set):
        return getattr(machine.environment.procedure_set, attr)
    elif attr in vars(core.Environment):
        return getattr(machine.environment, attr)

    elif attr in core.Tensor.dtype_names.keys():
        return core.Tensor.dtype_names[attr]
    elif attr in core.Tensor.dtype_short_names.keys():
        return core.Tensor.dtype_short_names[attr]

    raise NameError(attr)
