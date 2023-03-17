import mygrad
from mygrad.runtime import Runtime
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
        (out,) = Runtime.active.bind(cls, *args, **params)
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


class Neg(LLOp):
    @staticmethod
    def forward(x):
        return [-x]
