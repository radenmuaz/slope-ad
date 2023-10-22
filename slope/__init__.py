from slope import core

import os
import inspect

LOG_LRU = int(os.environ.get("LOG_LRU", 0))
LOG_JIT = int(os.environ.get("LOG_JIT", 0))
LOG_PYTREE = int(os.environ.get("LOG_PYTREE", 0))
LOG_ENV = int(os.environ.get("LOG_ENV", 0))
LOG_INIT = int(os.environ.get("LOG_INIT", 0))
INLINE_PROCEDURE = int(os.environ.get("INLINE_PROCEDURE", 0))


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
        dblog("Initializing slope.machine with", enable=LOG_INIT)
        dblog(inspect.getsource(slope_init), enable=LOG_INIT)
        machine = slope_init()
    return machine


def default_slope_init():
    from slope.environments.v1 import v1_environment

    return core.Machine(environment=v1_environment)


slope_init = default_slope_init


def set_slope_init(fn):
    global slope_init
    slope_init = fn


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
