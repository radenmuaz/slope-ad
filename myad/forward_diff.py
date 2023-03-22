import numpy as np
import myad
from myad import utils
from myad.runtime import Runtime
from myad.tracing import Trace, Tracer
from myad import pytrees


class JVPTracer(Tracer):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return self.get_aval(self.primal)


class JVPTrace(Trace):
    def pure(self, val):
        aval = Tracer.get_aval(val)
        zeros_like = np.zeros(aval.shape, aval.dtype)
        return JVPTracer(self, val, zeros_like)

    lift = pure

    def run_llop(self, llop, tracers, params):
        primals_in, tangents_in = utils.unzip2((t.primal, t.tangent) for t in tracers)
        primal_outs, tangent_outs = llop.jvp(primals_in, tangents_in, **params)
        return [JVPTracer(self, x, t) for x, t in zip(primal_outs, tangent_outs)]

def jvp_flat(f, primals, tangents):
    with myad.RT.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        outs = f(*tracers_in)
        tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
        primals_out, tangents_out = utils.unzip2(
            (t.primal, t.tangent) for t in tracers_out
        )
    return primals_out, tangents_out


def jvp(f, primals, tangents):
    primals_flat, in_tree = pytrees.tree_flatten(primals)
    tangents_flat, in_tree2 = pytrees.tree_flatten(tangents)
    if in_tree != in_tree2:
        raise TypeError
    f, out_tree = pytrees.flatten_fun(f, in_tree)
    primals_out_flat, tangents_out_flat = jvp_flat(f, primals_flat, tangents_flat)
    primals_out = pytrees.tree_unflatten(out_tree(), primals_out_flat)
    tangents_out = pytrees.tree_unflatten(out_tree(), tangents_out_flat)
    return primals_out, tangents_out


def add_one_to_a_scalar(scalar):
    assert np.ndim(scalar) == 0
    return 1 + scalar


# def jacfwd(f, x):
#     pushfwd = lambda v: jvp(f, (x,), (v,))[1]
#     vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
#     return vmap(pushfwd, (0,))(vecs_in)
