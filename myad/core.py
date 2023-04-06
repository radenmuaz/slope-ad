from contextlib import contextmanager
from typing import Type, Optional, Any, List
import operator
import itertools


from contextlib import contextmanager

import numpy as np

from typing import (
    Sequence,
    Callable,
    NamedTuple,
    Dict,
    Type,
    Hashable,
    Tuple,
    List,
    Any,
    Iterable,
    Iterator,
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
from myad import ops
from myad.tensor_shape import TensorShape
from myad.ops.base import Op
import itertools as it
import string
from functools import lru_cache


from typing import NamedTuple
from typing import Type, List, Tuple, Sequence, Optional, Any, DefaultDict
from typing import Callable, Type, Hashable, Dict, Iterable, Iterator
from typing import Union
from typing import Set
import string


class PPrint:
    lines: List[Tuple[int, str]]

    def __init__(self, lines):
        self.lines = lines

    def indent(self, indent: int) -> "PPrint":
        return PPrint([(indent + orig_indent, s) for orig_indent, s in self.lines])

    def __add__(self, rhs: "PPrint") -> "PPrint":
        return PPrint(self.lines + rhs.lines)

    def __rshift__(self, rhs: "PPrint") -> "PPrint":
        if not rhs.lines:
            return self
        if not self.lines:
            return rhs
        indent, s = self.lines[-1]
        indented_block = rhs.indent(indent + len(s))
        common_line = s + " " * rhs.lines[0][0] + rhs.lines[0][1]
        return PPrint(
            self.lines[:-1] + [(indent, common_line)] + indented_block.lines[1:]
        )

    def __str__(self) -> str:
        return "\n".join(" " * indent + s for indent, s in self.lines)

    def __repr__(self):
        return str(self)


def pp(s: Any) -> PPrint:
    return PPrint([(0, line) for line in str(s).splitlines()])


def vcat(ps: List[PPrint]) -> PPrint:
    return sum(ps, pp(""))



class Empty:
    pass


empty = Empty()



class Store:
    val = empty

    def set_value(self, val):
        assert self.val is empty
        self.val = val

    def __call__(self):
        return self.val


class NodeType(NamedTuple):
    name: str
    to_iterable: Callable
    from_iterable: Callable


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: Tuple["PyTreeDef", ...]

class Leaf:
    pass

leaf = Leaf()


def tree_flatten(x: Any) -> Any:
    def _tree_flatten(x: Any) -> Tuple[Iterable, Union[PyTreeDef, Leaf]]:
        node_type = myad.RT.node_types.get(type(x))
        if node_type:
            node_metadata, children = node_type.to_iterable(x)
            children_flat, child_trees = unzip2(mymap(_tree_flatten, children))
            flattened = itertools.chain.from_iterable(children_flat)
            return flattened, PyTreeDef(node_type, node_metadata, tuple(child_trees))
        else:
            return [x], leaf

    children_iter, treedef = _tree_flatten(x)
    return list(children_iter), treedef


def tree_unflatten(treedef: PyTreeDef, xs: List[Any]) -> Any:
    def _tree_unflatten(treedef: PyTreeDef, xs: Iterator) -> Any:
        if treedef is leaf:
            return next(xs)
        else:
            children = (_tree_unflatten(t, xs) for t in treedef.child_treedefs)
            return treedef.node_type.from_iterable(treedef.node_metadata, children)

    return _tree_unflatten(treedef, iter(xs))


def flatten_fun(f, in_tree):
    store = Store()

    def flat_fun(*args_flat):
        pytree_args = tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fun, store


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

    def run_op(self, op, tracers, params):
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
        return myad.RT.bind1(ops.Neg, self)

    def __add__(self, other):
        return myad.RT.bind1(ops.Add, self, other)

    def __radd__(self, other):
        return myad.RT.bind1(ops.Add, other, self)

    def __mul__(self, other):
        return myad.RT.bind1(ops.Mul, self, other)

    def __rmul__(self, other):
        return myad.RT.bind1(ops.Mul, other, self)

    def expand(self, shape, axes):
        return myad.RT.bind1(ops.Expand, self, shape, axes)
    def transpose(self, perm):
        return myad.RT.bind1(ops.Transpose, self, perm)

    def __bool__(self):
        return self.aval._bool(self)

    def __nonzero__(self):
        return self.aval._nonzero(self)

    @classmethod
    def get_aval(cls, x):
        if isinstance(x, cls):
            return x.aval
        elif type(x) in cls.TYPES:
            return Tensor.array(x)
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
        return Tensor.zeros(aval.shape, aval.dtype)

class EvalTracer(Tracer):
    def __init__(self, trace, val):
        self._trace = trace
        self.val = val

    def full_lower(self):
        return self.val

class EvalTrace(Trace):
    def pure(self, val):
        return EvalTracer(self, val)

    lift = pure

    def run_op(self, op, tracers, params):
        val_ins  = [t.val for t in tracers]
        eval_outs = op.eval(*val_ins, **params)
        return [EvalTracer(self, x,) for x in eval_outs]


from typing import Union

def mapped_aval(batch_dim, aval):
    shape = list(aval.shape)
    del shape[batch_dim]
    return TensorShape(tuple(shape), aval.dtype)

not_mapped = None

BatchAxis = Union[None, int]

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

    def run_op(self, op, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
        val_outs, bdim_outs = op.vmap(self.axis_size, vals_in, bdims_in, **params)
        return [BatchTracer(self, x, bd) for x, bd in myzip(val_outs, bdim_outs)]

    @property
    def axis_size(self):
        return self.main.global_data

def vmap_flat(f, in_axes, *args):
    def move_batch_axis(axis_size, src, dst, x):
        if src is not_mapped:
            target_shape = list(np.shape(x))
            target_shape.insert(dst, axis_size)
            return Tensor.broadcast_to(x, target_shape, [dst])
        elif src == dst:
            return x
        else:
            perm = [i for i in range(np.ndim(x)) if i != src]
            perm.insert(dst, src)
            return x.transpose(perm)
            # return moveaxis(x, src, dst)
    (axis_size,) = {x.shape[ax] for x, ax in myzip(args, in_axes) if ax is not not_mapped}
    with myad.RT.new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [
            BatchTracer(trace, x, ax) if ax is not None else x
            for x, ax in myzip(args, in_axes)
        ]
        outs = f(*tracers_in)
        tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
        vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_transposed = [
        move_batch_axis(axis_size, bdim, 0, val_out)
        for val_out, bdim in myzip(vals_out, bdims_out)
    ]
    return outs_transposed


def vmap(f, in_axes):
    def batched_f(*args):
        args_flat, in_tree = tree_flatten(args)
        in_axes_flat, in_tree2 = tree_flatten(in_axes)
        if in_tree != in_tree2:
            raise TypeError
        f_flat, out_tree = flatten_fun(f, in_tree)
        outs_flat = vmap_flat(f_flat, in_axes_flat, *args_flat)
        return tree_unflatten(out_tree(), outs_flat)

    return batched_f



# class EvalTrace(Trace):
#     pure = lift = lambda self, x: x

#     def run_op(self, op, tracers, params):
#         return op.eval(*tracers, **params)



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

    def run_op(self, op, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        primal_outs, tangent_outs = op.jvp(primals_in, tangents_in, **params)
        return [JVPTracer(self, x, t) for x, t in myzip(primal_outs, tangent_outs)]

def jvp_flat(f, primals, tangents):
    with myad.RT.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in myzip(primals, tangents)]
        outs = f(*tracers_in)
        tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
        primals_out, tangents_out = unzip2(
            (t.primal, t.tangent) for t in tracers_out
        )
    return primals_out, tangents_out


def jvp(f, primals, tangents):
    primals_flat, in_tree = tree_flatten(primals)
    tangents_flat, in_tree2 = tree_flatten(tangents)
    if in_tree != in_tree2:
        raise TypeError
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, tangents_out_flat = jvp_flat(f, primals_flat, tangents_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)
    tangents_out = tree_unflatten(out_tree(), tangents_out_flat)
    return primals_out, tangents_out


def add_one_to_a_scalar(scalar):
    assert np.ndim(scalar) == 0
    return 1 + scalar


def jacfwd(f, x):
    pushfwd = lambda v: jvp(f, (x,), (v,))[1]
    vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
    return vmap(pushfwd, (0,))(vecs_in)

class Var:
    val = None
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
# Atom = Union[Var, Lit, Tracer]


class JaxprEqn(NamedTuple):
    op: Op
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Atom]


class Jaxpr(NamedTuple):
    in_binders: Any
    eqns: List[JaxprEqn]
    outs: Any

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

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
        out_types = eqn.op.shape_eval(*in_types, **eqn.params)
        for out_binder, out_type in myzip(eqn.out_binders, out_types):
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

    mymap(write, jaxpr.in_binders, args)
    for eqn in jaxpr.eqns:
        in_vals = mymap(read, eqn.inputs)
        outs = myad.RT.bind(eqn.op, *in_vals, **eqn.params)
        mymap(write, eqn.out_binders, outs)
    return mymap(read, jaxpr.outs)


def jaxpr_as_fun(jaxpr: Jaxpr):
    return lambda *args: eval_jaxpr(jaxpr, args)


def split_list(lst: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    assert 0 <= n <= len(lst)
    return lst[:n], lst[n:]


def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(bs) == len(l)
    lst1: List[Any] = []
    lst2: List[Any] = []
    lists = lst1, lst2
    # lists = lst1: List[Any], lst2: List[Any] = list(), list()
    for b, x in myzip(bs, l):
        lists[b].append(x)
    return lst1, lst2


def var_str(names: DefaultDict[Var, str], v) -> str:
    return f"{names[v]}:{v.aval.str_short()}"


def pp_eqn(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    rule = eqn.op.pprint
    # if rule() is not None:
    #     return rule(names, eqn)
    # else:
    lhs = pp(" ".join(var_str(names, v) for v in eqn.out_binders))
    rhs = (
        pp(repr(eqn.op.__class__))
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

    def run_op(self, Op, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = Op.shape_eval(*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_eqn(JaxprEqn(Op, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data


class JaxprBuilder:
    eqns: List[JaxprEqn]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, Tracer]
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
        self, in_tracers: Any, out_tracers: Any
    ) -> Tuple[Jaxpr, List[Any]]:
        constvars, constvals = unzip2(self.constvals.items())
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
    literals = dict(zip(lit_binders, mymap(Lit, lit_vals)))
    new_eqns = [
        JaxprEqn(
            eqn.op,
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
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)

    builder = JaxprBuilder()
    with myad.RT.new_main(JaxprTrace, builder) as main:
        with myad.RT.new_dynamic(main):
            trace = JaxprTrace(main)
            tracers_in = [trace.new_arg(aval) for aval in avals_in]
            outs = f(*tracers_in)
            tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
            jaxpr, consts = builder.build(tracers_in, tracers_out)
    return jaxpr, consts, out_tree()


class Runtime:
    def __init__(self, root_trace=MainTrace(0, EvalTrace, None)):
        self.trace_stack: List[MainTrace] = []
        self.dynamic_trace: Optional[MainTrace] = None
        self.node_types = dict()
        self.trace_stack += [root_trace]

        self.node_types[tuple] = NodeType(
            str(tuple), lambda x: (None, x), lambda _, xs: tuple(xs)
        )
        self.node_types[list] = NodeType(
            str(list), lambda x: (None, x), lambda _, xs: list(xs)
        )
        self.node_types[dict] = NodeType(
            str(dict),
            lambda d: mymap(tuple, unzip2(sorted(d.items()))),
            lambda keys, vals: dict(zip(keys, vals)),
        )

    @contextmanager
    def new_main(self, trace_type: Type["Trace"], global_data=None):
        level = len(self.trace_stack)
        main = MainTrace(level, trace_type, global_data)
        self.trace_stack.append(main)

        try:
            yield main
        finally:
            self.trace_stack.pop()

    @contextmanager
    def new_dynamic(self, main: MainTrace):
        prev_dynamic_trace, self.dynamic_trace = self.dynamic_trace, main
        try:
            yield
        finally:
            self.dynamic_trace = prev_dynamic_trace

    def find_top_trace(self, xs) -> Trace:
        top_main = max(
            (x._trace.main for x in xs if isinstance(x, Tracer)),
            default=self.trace_stack[0],
            key=op.attrgetter("level"),
        )
        if self.dynamic_trace and self.dynamic_trace.level > top_main.level:
            top_main = self.dynamic_trace
        return top_main.trace_type(top_main)

    def full_lower(self, val: Any):
        if isinstance(val, Tracer):
            return val.full_lower()
        else:
            return val

    def full_raise(self, trace: Trace, val: Any) -> Tracer:
        if isinstance(val, list):
            breakpoint()
        if not isinstance(val, Tracer):
            return trace.pure(val)
        level = trace.main.level
        if val._trace.main is trace.main:
            return val
        elif val._trace.main.level < level:
            return trace.lift(val)
        elif val._trace.main.level > level:
            raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
        else:  # val._trace.level == level
            raise Exception(f"Different traces at same level: {val._trace}, {trace}.")

    def bind(self, op, *args, **params):
        top_trace = self.find_top_trace(args)
        tracers = [self.full_raise(top_trace, arg) for arg in args]
        outs = top_trace.run_op(op, tracers, params)
        lowered = [self.full_lower(out) for out in outs]
        return lowered

    def bind1(self, *args, **params):
        return self.bind(*args, **params)[0]


def swap(f):
    return lambda x, y: f(y, x)


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2




def mymap(f: Any, *xs: Any) -> Any:
    return list(map(f, *xs))




def myzip(*args: Any)-> Any:
    fst, *rest = args = mymap(list, args)
    n = len(fst)
    for arg in rest:
        assert len(arg) == n
    return list(zip(*args))
