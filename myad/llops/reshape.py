from myad.llops.base import LLOp
import numpy as np


class Reshape(LLOp):
    @staticmethod
    def forward(x, *, perm):
        return [np.reshape(x, perm)]
