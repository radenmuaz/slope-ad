from myad.ops.base import ReduceOp
from myad.tensor_shape import TensorShape
from typing import Tuple, List


class Sum(ReduceOp):
    @staticmethod
    def forward(x, *, axis):
        return [x.sum(axis)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]

    @staticmethod
    def shape_forward(x: TensorShape, *, axis: Tuple[int, ...]) -> List[TensorShape]:
        axis_ = set(axis)
        new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
        return [TensorShape(tuple(new_shape), x.dtype)]
