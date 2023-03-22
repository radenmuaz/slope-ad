from myad.llops.base import LLOp
from myad.array_shape import ArrayShape
from typing import Tuple, List, Sequence

import numpy as np


class Expand(LLOp):
    @staticmethod
    def forward(x, *, axis):
        return [np.broadcast_to(x, axis)]

    # @staticmethod
    # def jvp(primals, tangents):
    #     (x, y), (x_dot, y_dot) = primals, tangents
    #     return [x * y], [x_dot * y + x * y_dot]

    @staticmethod
    def shape_forward(
        x: ArrayShape, *, shape: Sequence[int], axes: Sequence[int]
    ) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]
