from mygrad.llops.base import LLOp
from mygrad.arrays import ShapedArray
from typing import Tuple, List

import numpy as np
class Sum(LLOp):
    @staticmethod
    def forward(x, *, axis):
        return [np.sum(x, axis)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]


    @staticmethod
    def forward_shape(
        x: ShapedArray, *, axis: Tuple[int, ...]
    ) -> List[ShapedArray]:
        axis_ = set(axis)
        new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
        return [ShapedArray(tuple(new_shape), x.dtype)]
