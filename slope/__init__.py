from slope import core


class LazyInitMachine:
    def __getattr__(self, attr):
        return getattr(M(), attr)


machine = LazyInitMachine()


class LazyInitEnv:
    def __getattr__(self, attr):
        return getattr(M().env, attr)


env = LazyInitEnv()
sev = env
numpy = env
torch = env


def M():
    global machine, env
    if type(machine) is LazyInitMachine:
        import inspect

        print("Initializing slope.machine with")
        print(inspect.getsource(slope_init))
        breakpoint()
        machine = slope_init()
        env = machine.env
    return machine


def default_slope_init():
    from slope.envs.v1 import v1_env

    return core.Machine(env=v1_env)


slope_init = default_slope_init


def set_slope_init(fn):
    global slope_init
    slope_init = fn


def __getattr__(attr):
    if (
        attr
        in "jvp vmap jit linearize vjp grad register_pytree_node tree_flatten tree_unflatten".split(
            " "
        )
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
