import numpy as np

from mygrad import pytrees as pt, runtime as reg, tracing as trc, utils


class JVPTracer(trc.Tracer):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return trc.get_aval(self.primal)


class JVPTrace(trc.Trace):
    pure = lift = lambda self, val: JVPTracer(self, val, trc.zeros_like(val))

    def run_llop(self, LLOp, tracers, params):
        primals_in, tangents_in = utils.unzip2((t.primal, t.tangent) for t in tracers)
        jvp_rule = reg.jvp_rules[LLOp]
        primal_outs, tangent_outs = jvp_rule(primals_in, tangents_in, **params)
        return [JVPTracer(self, x, t) for x, t in zip(primal_outs, tangent_outs)]


def jvp_v1(f, primals, tangents):
    with trc.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        out = f(*tracers_in)
        tracer_out = trc.full_raise(trace, out)
        primal_out, tangent_out = tracer_out.primal, tracer_out.tangent
    return primal_out, tangent_out


def jvp_flat(f, primals, tangents):
    with trc.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        outs = f(*tracers_in)
        tracers_out = [trc.full_raise(trace, out) for out in outs]
        primals_out, tangents_out = utils.unzip2(
            (t.primal, t.tangent) for t in tracers_out
        )
    return primals_out, tangents_out


def jvp(f, primals, tangents):
    primals_flat, in_tree = pt.tree_flatten(primals)
    tangents_flat, in_tree2 = pt.tree_flatten(tangents)
    if in_tree != in_tree2:
        raise TypeError
    f, out_tree = pt.flatten_fun(f, in_tree)
    primals_out_flat, tangents_out_flat = jvp_flat(f, primals_flat, tangents_flat)
    primals_out = pt.tree_unflatten(out_tree(), primals_out_flat)
    tangents_out = pt.tree_unflatten(out_tree(), tangents_out_flat)
    return primals_out, tangents_out


def add_one_to_a_scalar(scalar):
    assert np.ndim(scalar) == 0
    return 1 + scalar


# def jacfwd(f, x):
#     pushfwd = lambda v: jvp(f, (x,), (v,))[1]
#     vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
#     return vmap(pushfwd, (0,))(vecs_in)
