import mygrad
from mygrad.tracing import Runtime
import numpy as np
class LLOp:
    @staticmethod
    def forward(*args):
        raise NotImplementedError

    @staticmethod
    def vmap(*args):
        raise NotImplementedError

    @staticmethod
    def jvp(*args):
        raise NotImplementedError

    @classmethod
    def bind1(cls, *args, **params):
        (out,) = mygrad.RT.bind(cls, *args, **params)
        return out


class Identity(LLOp):
    @staticmethod
    def forward(x):
        return [x]

class Add(LLOp):
    @staticmethod
    def forward(x, y):
        return [x + y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]

class Mul(LLOp):
    @staticmethod
    def forward(x, y):
        return [x * y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]

class Sum(LLOp):
    @staticmethod
    def forward(x, *, axis):
        return [np.sum(x, axis)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]

class Transpose(LLOp):
    @staticmethod
    def forward(x, *, perm):
        return [np.transpose(x, perm)]

class Neg(LLOp):
    @staticmethod
    def forward(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]
