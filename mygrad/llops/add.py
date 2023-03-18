from mygrad.llops.base import LLOp


class Add(LLOp):
    @staticmethod
    def forward(x, y):
        return [x + y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]