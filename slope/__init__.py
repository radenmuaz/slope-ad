from slope import core
import os

SLOPE_DEBUG = os.environ.get("SLOPE_DEBUG", False)


class LazyInitMachine:
    def __getattr__(self, attr):
        return getattr(M(), attr)


machine = LazyInitMachine()


class LazyInitEnvironment:
    def __getattr__(self, attr):
        return getattr(M().environment, attr)


environment = LazyInitEnvironment()
sev = environment  # Slope EnVironment
numpy = environment
torch = environment


def M():
    global machine
    if type(machine) is LazyInitMachine:
        if SLOPE_DEBUG:
            import inspect

            print("Initializing slope.machine with")
            print(inspect.getsource(slope_init))
        machine = slope_init()
        global environment
        environment = machine.environment
    return machine


def default_slope_init():
    from slope.environments.v1 import v1_environment

    return core.Machine(environment=v1_environment)


slope_init = default_slope_init


def set_slope_init(fn):
    global slope_init
    slope_init = fn


def __getattr__(attr):
    if attr in (
        "jvp",
        "vmap",
        "jit",
        "linearize",
        "vjp",
        "grad",
        "register_pytree_node",
        "tree_flatten",
        "tree_unflatten",
    ):
        return getattr(machine, attr)
    else:
        return getattr(globals(), attr)


float32 = core.BaseArray.float32
float16 = core.BaseArray.float16
int8 = core.BaseArray.int8
int32 = core.BaseArray.int32
int64 = core.BaseArray.int64
uint8 = core.BaseArray.uint8
