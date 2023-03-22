from myad.tracing import Trace, Tracer
import myad
class EagerEvalTracer(Tracer):
    def __init__(self, trace, val):
        self._trace = trace
        self.val = val

    @property
    def aval(self):
        raise NotImplementedError

class EagerEvalTrace(Trace):
    def pure(self, val):
        return EagerEvalTracer(self, val)

    lift = pure

    def run_llop(self, llop, tracers, params):
        val_ins  = [t.val for t in tracers]
        eval_outs = llop.forward(*val_ins, **params)
        return [EagerEvalTracer(self, x,) for x in eval_outs]


# class EagerEvalTrace(Trace):
#     pure = lift = lambda self, x: x

#     def run_llop(self, llop, tracers, params):
#         return llop.forward(*tracers, **params)

