from slope.core import *
import slope as sp
from slope.core import *
from slope.opsets.v1 import v1_opset

rt = Runtime(opset=v1_opset)
# from slope.opsets.v1 import v1_opset

# global_rt = None

# def set_global_rt(rt):
#     global global_rt
#     global_rt = rt

# def get_global_rt(rt):
#     global global_rt
#     assert global_rt is not None, "set a Runtime first"
#     return global_rt

# def RT():
#     global RT_
#     if RT_ is None:
#         RT_ = Runtime(opset=v1_opset)
#     return RT_

# root_module/__init__.py

# import importlib

# # List of submodule names
# submodule_names = ['submodule1', 'submodule2', 'submodule3']

# # Dynamically import submodules and add their contents to the root module's namespace
# for submodule_name in submodule_names:
#     submodule = importlib.import_module(f'.{submodule_name}', package=__name__)
#     globals().update(vars(submodule))

# module.py

# def module_level_property(func):
#     result = func()
#     setattr(module, func.__name__, result)
#     return result

# def calculate_something():
#     return 42

# module_level_property(calculate_something)

# # Usage in another file
# import module

# print(module.calculate_something)  # This will print 42


# class ModulePropertyMeta(type):
#     def __getattr__(cls, name):
#         if name in cls._properties:
#             return cls._properties[name]()
#         raise AttributeError(f"module '{cls.__name__}' has no attribute '{name}'")

# class MyModule(metaclass=ModulePropertyMeta):
#     _properties = {}

# def module_property(func):
#     module_name = func.__module__
#     if module_name not in MyModule._properties:
#         MyModule._properties[module_name] = func
#     return func

# @module_property
# def module_level_property():
#     return "Hello from module level!"

# if __name__ == "__main__":
#     print(MyModule.module_level_property)  # Access the module-level "property" without ()

# class ModuleAttribute:
#     def __init__(self, func):
#         self.func = func

#     def __get__(self, instance, owner):
#         if instance is None:
#             return self.func(owner)
#         return self.func(instance)

# def my_module_function():
#     return 42

# module_var = ModuleAttribute(my_module_function)

# # Assume this code is in a file named "mymodule.py"

# @ModuleAttribute
# def module_function():
#     return 10

# # Usage in another module
# import mymodule

# print(mymodule.module_function)  # Prints the result of the function call, e.g., 10
