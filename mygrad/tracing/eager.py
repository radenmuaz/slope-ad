from mygrad.tracing.base import Trace
class EagerTrace(Trace):
    pure = lift = lambda self, x: x

    def run_llop(self, llop, tracers, params):
        return llop.forward(*tracers, **params)