from mygrad.llops.base import LLOp
import numpy as np
class Transpose(LLOp):
    @staticmethod
    def forward(x, *, perm):
        return [np.transpose(x, perm)]