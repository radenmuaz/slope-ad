from typing import NamedTuple
import numpy as np
import mygrad
class LLOp:
    @classmethod
    def forward(cls, *args):
        raise NotImplementedError
    @classmethod
    def reverse(cls, *args):
        raise NotImplementedError
    @classmethod
    def vmap(cls, *args):
        raise NotImplementedError
    @classmethod
    def jvp(cls, *args):
        raise NotImplementedError

    @classmethod
    def bind1(cls, *args, **params):
        (out,) = mygrad.RT.bind(cls, *args, **params)
        return out

class Add(LLOp):
    @classmethod
    def forward(cls, x, y):
        return [np.add(x, y)]

    @classmethod
    def reverse(cls, x, y):
        return [np.add(x, -y)]

class Mul(LLOp):
    @classmethod
    def forward(cls, x, y):
        return [np.multiply(x, y)]

    @classmethod
    def reverse(cls, x, y):
        return [np.divide(x, -y)]


class Neg(LLOp):
    @classmethod
    def forward(cls, x):
        return [np.negative(x)]

    @classmethod
    def reverse(cls, x):
        return [np.negative(x)]
