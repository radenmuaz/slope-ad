from slope.core import *
import slope as sp
from slope.core import *
from slope.opsets.v1 import v1_opset

RT_ = None


def RT():
    global RT_
    if RT_ is None:
        RT_ = Runtime(opset=v1_opset)
    return RT_


ops = RT().ops
procs = RT().procs


def __getattr__(name):
    for where in [ops, procs, globals()]:
        try:
            return getattr(where, name)
        except:
            pass
    raise AttributeError(f"{name} not found in slope")


# root_module/__init__.py

# import importlib

# # List of submodule names
# submodule_names = ['submodule1', 'submodule2', 'submodule3']

# # Dynamically import submodules and add their contents to the root module's namespace
# for submodule_name in submodule_names:
#     submodule = importlib.import_module(f'.{submodule_name}', package=__name__)
#     globals().update(vars(submodule))
