import mygrad
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