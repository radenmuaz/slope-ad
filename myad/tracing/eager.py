from myad.tracing.base import Trace


class EagerEvalTrace(Trace):
    pure = lift = lambda self, x: x

    def run_llop(self, llop, tracers, params):
        return llop.forward(*tracers, **params)
