from mygrad.llops.base import LLOp

class Neg(LLOp):
    @staticmethod
    def forward(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]