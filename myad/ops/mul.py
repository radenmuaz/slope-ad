from myad.ops.base import BinaryOp
from myad.tensor_shape import TensorShape
from typing import List

class Mul(BinaryOp):
    @staticmethod
    def forward(x, y):
        return [x * y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]

    @staticmethod
    def shape_forward(x: TensorShape, y: TensorShape) -> List[TensorShape]:
        if not isinstance(x, TensorShape) or not isinstance(y, TensorShape):
            raise TypeError
        if  TensorShape.from_numpy(x) !=  TensorShape.from_numpy(y):
            raise TypeError
        return [TensorShape(x.shape, x.dtype)]