from myad.llops.base import LLOp
from myad.array_shape import ArrayShape
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
    def shape_forward(x: ArrayShape, *, axis: Tuple[int, ...]) -> List[ArrayShape]:
        axis_ = set(axis)
        new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
        return [ArrayShape(tuple(new_shape), x.dtype)]
