# from slope.ad import Runtime, vmap, jvp, make_jaxpr, linearize, vjp, grad
from slope import ad

RT = ad.Runtime()

from typing import NamedTuple, Final
import numpy as np
from dataclasses import dataclass, asdict


# class DType(NamedTuple):
#     priority: int
#     itemsize: int
#     name: str
#     np: type

#     def __repr__(self):
#         return f"dtypes.{self.name}"


# bool: Final[DType] = DType(0, 1, "bool", bool)
# float16: Final[DType] = DType(0, 2, "half", np.float16)
# float32: Final[DType] = DType(4, 4, "float", np.float32)
# int8: Final[DType] = DType(0, 1, "char", np.int8)
# int32: Final[DType] = DType(1, 4, "int", np.int32)
# int64: Final[DType] = DType(2, 8, "int64", np.int64)
# uint8: Final[DType] = DType(0, 1, "uchar", np.uint8)


# def is_int(x: DType) -> bool:
#     return x in (int8, uint8, int32, int64)

# def is_float(x: DType) -> bool:
#     return x in (float16, float32)

# def is_unsigned(x: DType) -> bool:
#     return x in (uint8)
