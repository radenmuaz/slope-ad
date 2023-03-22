from contextlib import contextmanager
import operator as op

import numpy as np

from typing import (
    Type,
    List,
    Tuple,
    Sequence,
    Optional,
    Any,
    Union,
    NamedTuple,
    Dict,
    Set,
    DefaultDict,
    Callable,
)
from collections import defaultdict
import numpy as np
import operator as op

from myad.tensor import Tensor
import myad
from myad import llops, utils, pytrees
from myad.tensor_shape import TensorShape
from myad.llops.base import LLOp
import itertools as it
from myad.pretty_print import PPrint, pp, vcat
from myad import utils
from myad import pytrees
from myad.pytrees import PyTreeDef
import string
from functools import lru_cache


class MainTrace(NamedTuple):
    level: int
    trace_type: Type["Trace"]
    global_data: Optional[Any]


class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val):
        raise NotImplementedError

    def lift(self, val):
        raise NotImplementedError

    def run_llop(self, llop, tracers, params):
        raise NotImplementedError


class Tracer:
    TYPES = {
        bool,
        int,
        float,
        np.bool_,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        np.ndarray,
    }
    _trace: Trace

    __array_priority__ = 1000

    @property
    def aval(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return len(self.shape)

    def __neg__(self):
        return myad.RT.bind1(llops.Neg, self)

    def __add__(self, other):
        return myad.RT.bind1(llops.Add, self, other)

    def __radd__(self, other):
        return myad.RT.bind1(llops.Add, other, self)

    def __mul__(self, other):
        return myad.RT.bind1(llops.Mul, self, other)

    def __rmul__(self, other):
        return myad.RT.bind1(llops.Mul, other, self)

    def expand(self, shape, axes):
        return myad.RT.bind1(llops.Expand, self, shape, axes)
    def transpose(self, perm):
        return myad.RT.bind1(llops.Transpose, self, perm)

    def __bool__(self):
        return self.aval._bool(self)

    def __nonzero__(self):
        return self.aval._nonzero(self)

    @classmethod
    def get_aval(cls, x):
        if isinstance(x, cls):
            return x.aval
        # print(f"warn: {x} ({type(x)}) is not Tracer")
        elif type(x) in cls.TYPES:
            return Tensor(x)
        else:
            raise TypeError(x)

    def full_lower(self):
        return self  # default implementation

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def zeros_like(self, val):
        aval = self.get_aval(val)
        return np.zeros(aval.shape, aval.dtype)

class EagerEvalTracer(Tracer):
    def __init__(self, trace, val):
        self._trace = trace
        self.val = val

    def full_lower(self):
        return self.val

class EagerEvalTrace(Trace):
    def pure(self, val):
        return EagerEvalTracer(self, val)

    lift = pure

    def run_llop(self, llop, tracers, params):
        val_ins  = [t.val for t in tracers]
        eval_outs = llop.forward(*val_ins, **params)
        return [EagerEvalTracer(self, x,) for x in eval_outs]


from typing import Union

def mapped_aval(batch_dim, aval):
    shape = list(aval.shape)
    del shape[batch_dim]
    return TensorShape(tuple(shape), aval.dtype)


def move_batch_axis(axis_size, src, dst, x):
    if src is not_mapped:
        target_shape = list(np.shape(x))
        target_shape.insert(dst, axis_size)
        return x.expand(target_shape, [dst])
    elif src == dst:
        return x
    else:
        return moveaxis(x, src, dst)


def moveaxis(x, src: int, dst: int):
    perm = [i for i in range(np.ndim(x)) if i != src]
    perm.insert(dst, src)
    return x.transpose(perm)



class NotMapped:
    pass


not_mapped = NotMapped()

BatchAxis = Union[NotMapped, int]



class BatchTracer(Tracer):
    def __init__(self, trace, val, batch_dim: BatchAxis):
        self._trace = trace
        self.val = val
        self.batch_dim = batch_dim

    @property
    def aval(self):
        if self.batch_dim is not_mapped:
            return self.get_aval(self.val)
        else:
            return mapped_aval(self.batch_dim, self.get_aval(self.val))

    def full_lower(self):
        if self.batch_dim is not_mapped:
            return myad.RT.full_lower(self.val)
        else:
            return self


class BatchTrace(Trace):
    pure = lift = lambda self, val: BatchTracer(self, val, not_mapped)

    def run_llop(self, llop, tracers, params):
        vals_in, bdims_in = utils.unzip2((t.val, t.batch_dim) for t in tracers)
        val_outs, bdim_outs = llop.vmap(self.axis_size, vals_in, bdims_in, **params)
        return [BatchTracer(self, x, bd) for x, bd in zip(val_outs, bdim_outs)]

    @property
    def axis_size(self):
        return self.main.global_data

def vmap_flat(f, in_axes, *args):
    (axis_size,) = {x.shape[ax] for x, ax in zip(args, in_axes) if ax is not not_mapped}
    with myad.RT.new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [
            BatchTracer(trace, x, ax) if ax is not None else x
            for x, ax in zip(args, in_axes)
        ]
        outs = f(*tracers_in)
        tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
        vals_out, bdims_out = utils.unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_transposed = [
        move_batch_axis(axis_size, bdim, 0, val_out)
        for val_out, bdim in zip(vals_out, bdims_out)
    ]
    return outs_transposed


def vmap(f, in_axes):
    def batched_f(*args):
        args_flat, in_tree = pytrees.tree_flatten(args)
        in_axes_flat, in_tree2 = pytrees.tree_flatten(in_axes)
        if in_tree != in_tree2:
            raise TypeError
        f_flat, out_tree = pytrees.flatten_fun(f, in_tree)
        outs_flat = vmap_flat(f_flat, in_axes_flat, *args_flat)
        return pytrees.tree_unflatten(out_tree(), outs_flat)

    return batched_f



# class EagerEvalTrace(Trace):
#     pure = lift = lambda self, x: x

#     def run_llop(self, llop, tracers, params):
#         return llop.forward(*tracers, **params)



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


def jacfwd(f, x):
    pushfwd = lambda v: jvp(f, (x,), (v,))[1]
    vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
    return vmap(pushfwd, (0,))(vecs_in)

class Var:
    aval: TensorShape

    def __init__(self, aval):
        self.aval = aval


class Lit:
    val: Any
    aval: TensorShape

    def __init__(self, val):
        self.aval = aval = TensorShape.from_numpy(Tracer.get_aval(val))
        self.val = np.array(val, aval.dtype)


Atom = Union[Var, Lit]


class JaxprEqn(NamedTuple):
    llop: LLOp
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
        in_binders = ", ".join(var_str(names, x) for x in self.in_binders)
        eqns = vcat([pp_eqn(names, e) for e in self.eqns])
        outs = ", ".join(
            names[v] if isinstance(v, Var) else str(v.val) for v in self.outs
        )
        return str(
            pp(f"{{ lambda {in_binders} .")
            + ((pp("let ") >> eqns) + pp(f"in ( {outs} ) }}")).indent(2)
        )


class JaxprType(NamedTuple):
    in_types: List[TensorShape]
    out_types: List[TensorShape]

    def __repr__(self):
        in_types = ", ".join(aval.str_short() for aval in self.in_types)
        out_types = ", ".join(aval.str_short() for aval in self.out_types)
        return f"({in_types}) -> ({out_types})"


def typecheck_jaxpr(jaxpr: Jaxpr) -> JaxprType:
    env: Set[Var] = set()

    for v in jaxpr.in_binders:
        if v in env:
            raise TypeError
        env.add(v)

    for eqn in jaxpr.eqns:
        in_types = [typecheck_atom(env, x) for x in eqn.inputs]
        out_types = eqn.llop.shape_forward(*in_types, **eqn.params)
        for out_binder, out_type in zip(eqn.out_binders, out_types):
            if not out_type == out_binder.aval:
                raise TypeError
        for out_binder in eqn.out_binders:
            if out_binder in env:
                raise TypeError
            env.add(out_binder)

    in_types = [v.aval for v in jaxpr.in_binders]
    out_types = [typecheck_atom(env, x) for x in jaxpr.outs]
    return JaxprType(in_types, out_types)


def typecheck_atom(env: Set[Var], x: Atom) -> TensorShape:
    if isinstance(x, Var):
        if x not in env:
            raise TypeError("unbound variable")
        return x.aval
    elif isinstance(x, Lit):
        return TensorShape.from_numpy(Tracer.get_aval(x.val))
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
        outs = myad.RT.bind(eqn.LLOp, *in_vals, **eqn.params)
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
    return f"{names[v]}:{v.aval.str_short()}"


def pp_eqn(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    rule = eqn.llop.pprint
    # if rule() is not None:
    #     return rule(names, eqn)
    # else:
    lhs = pp(" ".join(var_str(names, v) for v in eqn.out_binders))
    rhs = (
        pp(repr(eqn.llop.__class__))
        >> pp_params(eqn.params)
        >> pp(
            " ".join(names[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs)
        )
    )
    return lhs >> pp(" = ") >> rhs


def pp_params(params: Dict[str, Any]) -> PPrint:
    items = sorted(params.items())
    if items:
        return pp(" [ ") >> vcat([pp(f"{k}={v}") for k, v in items]) >> pp(" ] ")
    else:
        return pp(" ")


class JaxprTracer(Tracer):
    __slots__ = ["aval"]
    aval: TensorShape

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval


class JaxprTrace(Trace):
    def new_arg(self, aval: TensorShape) -> JaxprTracer:
        aval = TensorShape.from_numpy(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)

        return tracer

    def get_or_make_const_tracer(self, val: Any) -> JaxprTracer:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(
                self, TensorShape.from_numpy(Tracer.get_aval(val))
            )
            self.builder.add_const(tracer, val)
        return tracer

    pure = lift = get_or_make_const_tracer

    def run_llop(self, LLOp, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = LLOp.shape_forward(*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_eqn(JaxprEqn(LLOp, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data


class JaxprBuilder:
    eqns: List[JaxprEqn]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, JaxprTracer]
    constvals: Dict[Var, Any]
    tracers: List[JaxprTracer]

    def __init__(self):
        self.eqns = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    def new_tracer(self, trace: JaxprTrace, aval: TensorShape) -> JaxprTracer:
        tracer = JaxprTracer(trace, aval)
        self.tracers.append(tracer)
        return tracer

    def add_eqn(self, eqn: JaxprEqn) -> None:
        self.eqns.append(eqn)

    def add_var(self, tracer: JaxprTracer) -> Var:
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
        return var

    def getvar(self, tracer: JaxprTracer) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer: JaxprTracer, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(
        self, in_tracers: List[JaxprTracer], out_tracers: List[JaxprTracer]
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
            eqn.llop,
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


@lru_cache()
def make_jaxpr(
    f: Callable,
    *avals_in: TensorShape,
) -> Tuple[Jaxpr, List[Any], PyTreeDef]:
    avals_in, in_tree = pytrees.tree_flatten(avals_in)
    f, out_tree = pytrees.flatten_fun(f, in_tree)

    builder = JaxprBuilder()
    with myad.RT.new_main(JaxprTrace, builder) as main:
        with myad.RT.new_dynamic(main):
            trace = JaxprTrace(main)
            tracers_in = [trace.new_arg(aval) for aval in avals_in]
            outs = f(*tracers_in)
            tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
            jaxpr, consts = builder.build(tracers_in, tracers_out)
    return jaxpr, consts, out_tree()
