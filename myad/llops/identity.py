from myad.llops.base import LLOp


class Identity(LLOp):
    @staticmethod
    def forward(x):
        return [x]
