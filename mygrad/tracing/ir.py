from typing import Type, List, Tuple, Sequence, Optional, Any, Union, NamedTuple, Dict
from collections import defaultdict
from mygrad import llops
import numpy as np
import operator as op

from mygrad.arrays import ShapedArray
from mygrad.tracing.base import Tracer, Trace
from mygrad import utils
from mygrad.pretty_print import pp, vcat
import string
import itertools as it
from typing import DefaultDict
class Var:
    aval: ShapedArray

    def __init__(self, aval):
        self.aval = aval


class Lit:
    val: Any
    aval: ShapedArray

    def __init__(self, val):
        self.aval = aval = ShapedArray.raise_to_shaped(Tracer.get_aval(val))
        self.val = np.array(val, aval.dtype)


Atom = Union[Var, Lit]


class JaxprEqn(NamedTuple):
    LLOp: llops.LLOp
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Var]


class Jaxpr(NamedTuple):
    in_binders: List[Var]
    eqns: List[JaxprEqn]
    outs: List[Atom]

    def __hash__(self):
        return id(self)

    __eq__ = op.is_

    def __repr__(self):
        namegen = (
            "".join(s)
            for r in it.count(1)
            for s in it.permutations(string.ascii_lowercase, r)
        )
        names = defaultdict(lambda: next(namegen))
        in_binders = ", ".join(var_str(names, x) for x in jaxpr.in_binders)
        eqns = vcat([pp_eqn(names, e) for e in jaxpr.eqns])
        outs = ", ".join(
            names[v] if isinstance(v, Var) else str(v.val) for v in jaxpr.outs
        )
        return pp(f"{{ lambda {in_binders} .") + (
            (pp("let ") >> eqns) + pp(f"in ( {outs} ) }}")
        ).indent(2)


class Jaxpmygrad.RType(NamedTuple):
    in_types: List[ShapedArray]
    out_types: List[ShapedArray]

    def __repr__(self):
        in_types = ", ".join(aval.str_shomygrad.RT() for aval in self.in_types)
        out_types = ", ".join(aval.str_shomygrad.RT() for aval in self.out_types)
        return f"({in_types}) -> ({out_types})"


def typecheck_jaxpr(jaxpr: Jaxpr) -> Jaxpmygrad.RType:
    env: Set[Var] = set()

    for v in jaxpr.in_binders:
        if v in env:
            raise TypeError
        env.add(v)

    for eqn in jaxpr.eqns:
        in_types = [typecheck_atom(env, x) for x in eqn.inputs]
        out_types = forward_shape_rules[eqn.LLOp](*in_types, **eqn.params)
        for out_binder, out_type in zip(eqn.out_binders, out_types):
            if not out_type == out_binder.aval:
                raise TypeError
        for out_binder in eqn.out_binders:
            if out_binder in env:
                raise TypeError
            env.add(out_binder)

    in_types = [v.aval for v in jaxpr.in_binders]
    out_types = [typecheck_atom(env, x) for x in jaxpr.outs]
    return Jaxpmygrad.RType(in_types, out_types)


def typecheck_atom(env: Set[Var], x: Atom) -> ShapedArray:
    if isinstance(x, Var):
        if x not in env:
            raise TypeError("unbound variable")
        return x.aval
    elif isinstance(x, Lit):
        return ShapedArray.raise_to_shaped(Tracer.get_aval(x.val))
    else:
        assert False


def eval_jaxpr(jaxpr: Jaxpr, args: List[Any]) -> List[Any]:
    env: Dict[Var, Any] = {}

    def read(x: Atom) -> Any:
        return env[x] if type(x) is Var else x.val

    def write(v: Var, val: Any) -> None:
        assert v not in env  # single-assignment
        env[v] = val

    map(write, jaxpr.in_binders, args)
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.inputs)
        outs = bind(eqn.LLOp, *in_vals, **eqn.params)
        map(write, eqn.out_binders, outs)
    return map(read, jaxpr.outs)


def jaxpr_as_fun(jaxpr: Jaxpr):
    return lambda *args: eval_jaxpr(jaxpr, args)


def split_list(lst: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    assert 0 <= n <= len(lst)
    return lst[:n], lst[n:]


def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(bs) == len(l)
    lists = lst1, lst2 = [], []
    for b, x in zip(bs, l):
        lists[b].append(x)
    return lst1, lst2


def var_str(names: DefaultDict[Var, str], v: Var) -> str:
    return f'{names[v]}:{v.aval.str_shomygrad.mygrad.RT()}'

def pp_eqn(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    rule = pp_rules.get(eqn.LLOp)
    if rule:
        return rule(names, eqn)
    else:
        lhs = pp(' '.join(var_str(names, v) for v in eqn.out_binders))
        rhs = (pp(eqn.LLOp.name) >> pp_params(eqn.params) >>
           pp(' '.join(names[x] if isinstance(x, Var) else str(x.val)
                       for x in eqn.inputs)))
        return lhs >> pp(' = ') >> rhs

def pp_params(params: Dict[str, Any]) -> PPrint:
    items = sorted(params.items())
    if items:
        return pp(' [ ') >> vcat([pp(f'{k}={v}') for k, v in items]) >> pp(' ] ')
    else:
        return pp(' ')


class JaxprTracer(Tracer):
    __slots__ = ["aval"]
    aval: ShapedArray

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval


class JaxprTrace(Trace):
    def new_arg(self, aval: ShapedArray) -> Jaxpmygrad.RTracer:
        aval = ShapedArray.raise_to_shaped(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)
        return tracer

    def get_or_make_const_tracer(self, val: Any) -> Jaxpmygrad.RTracer:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, ShapedArray.raise_to_shaped(Tracer.get_aval(val)))
            self.builder.add_const(tracer, val)
        return tracer

    pure = lift = get_or_make_const_tracer

    def run_llop(self, LLOp, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = forward_shape_rules[LLOp](*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_eqn(JaxprEqn(LLOp, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data


forward_shape_rules = {}


class JaxprBuilder:
    eqns: List[JaxprEqn]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, Jaxpmygrad.RTracer]
    constvals: Dict[Var, Any]
    tracers: List[Jaxpmygrad.RTracer]

    def __init__(self):
        self.eqns = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    def new_tracer(self, trace: Jaxpmygrad.RTrace, aval: ShapedArray) -> Jaxpmygrad.RTracer:
        tracer = Jaxpmygrad.RTracer(trace, aval)
        self.tracers.append(tracer)
        return tracer

    def add_eqn(self, eqn: JaxprEqn) -> None:
        self.eqns.append(eqn)

    def add_var(self, tracer: Jaxpmygrad.RTracer) -> Var:
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
        return var

    def getvar(self, tracer: Jaxpmygrad.RTracer) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer: Jaxpmygrad.RTracer, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(
        self, in_tracers: List[Jaxpmygrad.RTracer], out_tracers: List[Jaxpmygrad.RTracer]
    ) -> Tuple[Jaxpr, List[Any]]:
        constvars, constvals = utils.unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        jaxpr = Jaxpr(in_binders, self.eqns, out_vars)
        typecheck_jaxpr(jaxpr)
        jaxpr, constvals = _inline_literals(jaxpr, constvals)
        return jaxpr, constvals


def _inline_literals(jaxpr: Jaxpr, consts: List[Any]) -> Tuple[Jaxpr, List[Any]]:
    const_binders, other_binders = split_list(jaxpr.in_binders, len(consts))
    scalars = [type(x) in Tracer.TYPES and not Tracer.get_aval(x).shape for x in consts]
    new_const_binders, lit_binders = partition_list(scalars, const_binders)
    new_consts, lit_vals = partition_list(scalars, consts)
    literals = dict(zip(lit_binders, map(Lit, lit_vals)))
    new_eqns = [
        JaxprEqn(
            eqn.LLOp,
            [literals.get(x, x) for x in eqn.inputs],
            eqn.params,
            eqn.out_binders,
        )
        for eqn in jaxpr.eqns
    ]
    new_outs = [literals.get(x, x) for x in jaxpr.outs]
    new_jaxpr = Jaxpr(new_const_binders + other_binders, new_eqns, new_outs)
    typecheck_jaxpr(new_jaxpr)
    return new_jaxpr, new_consts



# @lru_cache()  # ShapedArrays are hashable
# def make_jaxpr_v1(f, *avals_in):
#     avals_in, in_tree = tree_flatten(avals_in)
#     f, out_tree = flatten_fun(f, in_tree)

#     builder = JaxprBuilder()
#     with new_main(Jaxpmygrad.RTrace, builder) as main:
#         trace = Jaxpmygrad.RTrace(main)
#         tracers_in = [trace.new_arg(aval) for aval in avals_in]
#         outs = f(*tracers_in)
#         tracers_out = [full_raise(trace, out) for out in outs]
#         jaxpr, consts = builder.build(tracers_in, tracers_out)
#     return jaxpr, consts, out_tree()
