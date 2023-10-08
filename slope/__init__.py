from slope import core
import os

SLOPE_DEBUG = int(os.environ.get("SLOPE_DEBUG", 0))


def dblog(*msg, level=0):
    if SLOPE_DEBUG >= level:
        print(*msg)


class LazyInitMachine:
    def __getattr__(self, attr):
        return getattr(M(), attr)


machine = LazyInitMachine()


def M():
    global machine
    if type(machine) is LazyInitMachine:
        if SLOPE_DEBUG > 0:
            import inspect

            print("Initializing slope.machine with")
            print(inspect.getsource(slope_init))
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
    if attr in vars(core.Machine):
        return getattr(machine, attr)
    M()
    if attr in vars(machine.environment.operator_set):
        return getattr(machine.environment.operator_set, attr)
    elif attr in vars(machine.environment.procedure_set):
        return getattr(machine.environment.procedure_set, attr)
    elif attr in [a for a in dir(machine.environment) if a[:2] != "__"]:
        return getattr(machine.environment, attr)
    elif attr in core.Tensor.dtype_names.keys():
        return core.Tensor.dtype_names[attr]
    elif attr in core.Tensor.dtype_short_names.keys():
        return core.Tensor.dtype_short_names[attr]
    elif attr in (globals_dict := globals()):
        return globals_dict[attr]
    raise NameError(attr)
