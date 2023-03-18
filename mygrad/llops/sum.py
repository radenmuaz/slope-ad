from mygrad.llops.base import LLOp
import numpy as np
class Sum(LLOp):
    @staticmethod
    def forward(x, *, axis):
        return [np.sum(x, axis)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]
