from myad.ops.base import ShapeOp
import numpy as np


class Reshape(ShapeOp):
    @staticmethod
    def forward(x, *, perm):
        return [np.reshape(x, perm)]
