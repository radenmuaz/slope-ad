from contextlib import contextmanager
import operator

from contextlib import contextmanager

import numpy as np
import itertools

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
    Optional,
    Any,
    Union,
    NamedTuple,
    Dict,
    Set,
    DefaultDict,
    Callable,
    Hashable,
)
from collections import defaultdict
import numpy as np
import operator as op
import string
from functools import lru_cache

import myad
from myad.array_shape import ArrayShape, ValuedArrayShape
from myad import ops

def full_like(self, val, fill_val):
    return np.full(val.shape, fill_val, val.dtype)

def ones_like(self, val):
    return np.ones(val.shape, val.dtype)

def zeros_like(self, val):
    return np.zeros(val.shape, val.dtype)

def swap(f):
    return lambda x, y: f(y, x)


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


def list_map(f: Any, *xs: Any) -> Any:
    return list(map(f, *xs))


def list_zip(*args: Any) -> Any:
    fst, *rest = args = list_map(list, args)
    n = len(fst)
    for arg in rest:
        assert len(arg) == n
    return list(zip(*args))


def split_half(lst: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert not len(lst) % 2
    return split_list(lst, len(lst) // 2)


def merge_lists(which: List[bool], l1: List[Any], l2: List[Any]) -> List[Any]:
    l1, l2 = iter(l1), iter(l2)
    out = [next(l2) if b else next(l1) for b in which]
    assert next(l1, None) is next(l2, None) is None
    return out

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
            children_flat, child_trees = unzip2(list_map(_tree_flatten, children))
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
        return ops.add(self, other)

    def __radd__(self, other):
        return ops.add(other, self)
    
    def __sub__(self, other):
        return ops.sub(self, other)

    def __rsub__(self, other):
        return ops.sub(other, self)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __rmul__(self, other):
        return ops.mul(other, self)

    def __div__(self, other):
        return ops.div(self, other)
    
    def __rdiv__(self, other):
        return ops.div(other, self)
    
    def __truediv__(self, other):
        return ops.div(self, other)

    def __rtruediv__(self, other):
        return ops.div(other, self)

    def __pow__(self, other):
        return ops.pow(self, other)

    # def transpose(self, perm):
    #     return myad.RT.bind1(ops.Transpose, self, perm)

    def __bool__(self):
        return self.aval._bool(self)

    def __nonzero__(self):
        return self.aval._nonzero(self)

    # @classmethod
    # def get_aval(cls, x):
    #     if isinstance(x, cls):
    #         return x.aval
    #     elif type(x) in cls.TYPES:
    #         return ValuedArrayShape(np.array(x))
    #     else:
    #         raise TypeError(x)
    
    @staticmethod
    def get_aval(x):
        if isinstance(x, Tracer):
            return x.aval
        elif type(x) in Tracer.TYPES:
            return ValuedArrayShape(np.array(x))
        else:
            raise TypeError(x)

    def full_lower(self):
        return self  # default implementation

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


class EvalTrace(Trace):
    pure = lift = lambda self, x: x  # no boxing in Tracers needed

    def run_op(self, op, tracers, params):
        # 
        return op.eval(*tracers, **params)

# class EvalTracer(Tracer):
#     def __init__(self, trace, val):
#         self._trace = trace
#         self.val = val

#     def full_lower(self):
#         return self.val


# class EvalTrace(Trace):
#     def pure(self, val):
#         return EvalTracer(self, val)

#     lift = pure

#     def run_op(self, op, tracers, params):
#         val_ins = [t.val for t in tracers]
#         eval_outs = op.eval(*val_ins, **params)
#         return [
#             EvalTracer(
#                 self,
#                 x,
#             )
#             for x in eval_outs
#         ]


from typing import Union


def mapped_aval(batch_dim, aval):
    shape = list(aval.shape)
    del shape[batch_dim]
    return ArrayShape(tuple(shape), aval.dtype)


BatchAxis = Union[None, int]


class BatchTracer(Tracer):
    def __init__(self, trace, val, batch_dim: BatchAxis):
        self._trace = trace
        self.val = val
        self.batch_dim = batch_dim

    @property
    def aval(self):
        if self.batch_dim is None:
            return self.get_aval(self.val)
        else:
            return mapped_aval(self.batch_dim, self.get_aval(self.val))

    def full_lower(self):
        if self.batch_dim is None:
            return myad.RT.full_lower(self.val)
        else:
            return self


class BatchTrace(Trace):
    pure = lift = lambda self, val: BatchTracer(self, val, None)

    def run_op(self, op, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
        val_outs, bdim_outs = op.vmap(self.axis_size, vals_in, bdims_in, **params)
        return [BatchTracer(self, x, bd) for x, bd in list_zip(val_outs, bdim_outs)]

    @property
    def axis_size(self):
        return self.main.global_data

def move_batch_axis(axis_size, src, dst, x):
    if src is None:
        target_shape = list(x.shape)
        target_shape.insert(dst, axis_size)
        out_ndim = len(target_shape)
        if type(dst) in (tuple, list):
            out_ndim += 1
        reshape_shape = [1 if ax==dst else target_shape for ax in range(out_ndim)]
        x = ops.Reshape.do(x, reshape_shape)
        x = ops.Broadcast.do(x, target_shape)
        return x
    elif src == dst:
        return x
    else:
        perm = [i for i in range(np.ndim(x)) if i != src]
        perm.insert(dst, src)
        return ops.Transpose.do(x, perm)

def vmap_flat(f, in_axes, *args):
    (axis_size,) = {
        x.shape[ax] for x, ax in list_zip(args, in_axes) if ax is not None
    }
    with myad.RT.new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [
            BatchTracer(trace, x, ax) if ax is not None else x
            for x, ax in list_zip(args, in_axes)
        ]
        outs = f(*tracers_in)
        tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
        vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_transposed = [
        move_batch_axis(axis_size, bdim, 0, val_out)
        for val_out, bdim in list_zip(vals_out, bdims_out)
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
        return [JVPTracer(self, x, t) for x, t in list_zip(primal_outs, tangent_outs)]


def jvp_flat(f, primals, tangents):
    with myad.RT.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in list_zip(primals, tangents)]
        outs = f(*tracers_in)
        tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
        primals_out, tangents_out = unzip2((t.primal, t.tangent) for t in tracers_out)
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



def jacfwd(f, x):
    pushfwd = lambda v: jvp(f, (x,), (v,))[1]
    vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
    return vmap(pushfwd, (0,))(vecs_in)


class Var:
    val = None
    aval: ArrayShape

    def __init__(self, aval):
        self.aval = aval


class Lit:
    val: Any
    aval: ArrayShape

    def __init__(self, val):
        self.aval = aval = ArrayShape.like(Tracer.get_aval(val))
        self.val = np.array(val, aval.dtype)


Atom = Union[Var, Lit]


class JaxprEqn(NamedTuple):
    op: ops.Op
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
            for r in itertools.count(1)
            for s in itertools.permutations(string.ascii_lowercase, r)
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
    in_types: List[ArrayShape]
    out_types: List[ArrayShape]

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
        for out_binder, out_type in list_zip(eqn.out_binders, out_types):
            if not out_type == out_binder.aval:
                raise TypeError
        for out_binder in eqn.out_binders:
            if out_binder in env:
                raise TypeError
            env.add(out_binder)

    in_types = [v.aval for v in jaxpr.in_binders]
    out_types = [typecheck_atom(env, x) for x in jaxpr.outs]
    return JaxprType(in_types, out_types)


def typecheck_atom(env: Set[Var], x: Atom) -> ArrayShape:
    if isinstance(x, Var):
        if x not in env:
            raise TypeError("unbound variable")
        return x.aval
    elif isinstance(x, Lit):
        return Tracer.get_aval(x.val)
    else:
        assert False


def eval_jaxpr(jaxpr: Jaxpr, args: List[Any]) -> List[Any]:
    env: Dict[Var, Any] = {}

    def read(x: Atom) -> Any:
        return env[x] if type(x) is Var else x.val

    def write(v: Var, val: Any) -> None:
        assert v not in env  # single-assignment
        env[v] = val

    list_map(write, jaxpr.in_binders, args)
    for eqn in jaxpr.eqns:
        in_vals = list_map(read, eqn.inputs)
        outs = myad.RT.bind(eqn.op, *in_vals, **eqn.params)
        list_map(write, eqn.out_binders, outs)
    return list_map(read, jaxpr.outs)


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
    for b, x in list_zip(bs, l):
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
    aval: ArrayShape

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval


class JaxprTrace(Trace):
    def new_arg(self, aval) -> JaxprTracer:
        aval = ArrayShape.like(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)

        return tracer

    def get_or_make_const_tracer(self, val: Any) -> JaxprTracer:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, Tracer.get_aval(val))
            self.builder.add_const(tracer, val)
        return tracer

    pure = lift = get_or_make_const_tracer

    def run_op(self, op, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = op.shape_eval(*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_eqn(JaxprEqn(op, inputs, params, outvars))
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

    def new_tracer(self, trace: JaxprTrace, aval: ArrayShape) -> JaxprTracer:
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

    def build(self, in_tracers: Any, out_tracers: Any) -> Tuple[Jaxpr, List[Any]]:
        constvars, constvals = unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        jaxpr = Jaxpr(in_binders, self.eqns, out_vars)
        typecheck_jaxpr(jaxpr)
        jaxpr, constvals = self._inline_literals(jaxpr, constvals)
        return jaxpr, constvals


    def _inline_literals(self, jaxpr: Jaxpr, consts: List[Any]) -> Tuple[Jaxpr, List[Any]]:
        const_binders, other_binders = split_list(jaxpr.in_binders, len(consts))
        scalars = [type(x) in Tracer.TYPES and not Tracer.get_aval(x).shape for x in consts]
        new_const_binders, lit_binders = partition_list(scalars, const_binders)
        new_consts, lit_vals = partition_list(scalars, consts)
        literals = dict(zip(lit_binders, list_map(Lit, lit_vals)))
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
    *avals_in: ArrayShape,
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




def linearize_flat(f, *primals_in):
    pvals_in = [PartialVal.known(x) for x in primals_in] + [
        PartialVal.unknown(ArrayShape.like(Tracer.get_aval(x))) for x in primals_in
    ]

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    jaxpr, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)
    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    f_lin = lambda *tangents: eval_jaxpr(jaxpr, [*consts, *tangents])
    return primals_out, f_lin


def linearize(f, *primals_in):
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, f_lin_flat = linearize_flat(f, *primals_in_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)

    def f_lin(*tangents_in):
        tangents_in_flat, in_tree2 = tree_flatten(tangents_in)
        if in_tree != in_tree2:
            raise TypeError
        tangents_out_flat = f_lin_flat(*tangents_in_flat)
        return tree_unflatten(out_tree(), tangents_out_flat)

    return primals_out, f_lin


class PartialVal(NamedTuple):
    aval: ArrayShape
    const: Optional[Any]

    @classmethod
    def known(cls, val: Any):
        return PartialVal(Tracer.get_aval(val), val)

    @classmethod
    def unknown(cls, aval: ArrayShape):
        return PartialVal(aval, None)

    is_known = property(lambda self: self.const is not None)
    is_unknown = property(lambda self: self.const is None)


def partial_eval_flat(
    f: Callable, pvals_in: List[PartialVal]
) -> Tuple[Jaxpr, List[PartialVal], List[Any]]:
    with myad.RT.new_main(PartialEvalTrace) as main:
        trace = PartialEvalTrace(main)
        tracers_in = [trace.new_arg(pval) for pval in pvals_in]
        outs = f(*tracers_in)
        tracers_out = [myad.RT.full_raise(trace, out) for out in outs]
        pvals_out = [t.pval for t in tracers_out]
        unk_tracers_in = [t for t in tracers_in if t.pval.is_unknown]
        unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
        jaxpr, consts = tracers_to_jaxpr(unk_tracers_in, unk_tracers_out)

    return jaxpr, pvals_out, consts


from weakref import ref, ReferenceType


class LambdaBindingRecipe(NamedTuple):
    pass


class ConstRecipe(NamedTuple):
    val: Any


class JaxprEqnRecipe(NamedTuple):
    prim: ops.Op
    tracers_in: List["PartialEvalTracer"]
    params: Dict[str, Any]
    avals_out: List[ArrayShape]
    tracer_refs_out: List["ReferenceType[PartialEvalTracer]"]


JaxprRecipe = Union[LambdaBindingRecipe, ConstRecipe, JaxprEqnRecipe]


class PartialEvalTracer(Tracer):
    pval: PartialVal
    recipe: Optional[JaxprRecipe]

    def __init__(self, trace, pval, recipe):
        self._trace = trace
        self.pval = pval
        self.recipe = recipe

    aval = property(lambda self: self.pval.aval)

    def full_lower(self):
        if self.pval.is_known:
            return myad.RT.full_lower(self.pval.const)
        return self


class PartialEvalTrace(Trace):
    def new_arg(self, pval: PartialVal) -> Any:
        return PartialEvalTracer(self, pval, LambdaBindingRecipe())

    def lift(self, val: Any) -> PartialEvalTracer:
        return PartialEvalTracer(self, PartialVal.known(val), None)

    pure = lift

    def instantiate_const(self, tracer: PartialEvalTracer) -> PartialEvalTracer:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = PartialVal.unknown(ArrayShape.like(tracer.aval))
            return PartialEvalTracer(self, pval, ConstRecipe(tracer.pval.const))

    def run_op(self, op, tracers, params):
        if all(t.pval.is_known for t in tracers):
            return myad.RT.bind(op, *map(myad.RT.full_lower, tracers), **params)
        tracers_in = [self.instantiate_const(t) for t in tracers]
        avals_in = [t.aval for t in tracers_in]
        avals_out = op.shape_eval(*avals_in, **params)
        tracers_out = [
            PartialEvalTracer(self, PartialVal.unknown(aval), None)
            for aval in avals_out
        ]
        eqn = JaxprEqnRecipe(op, tracers_in, params, avals_out, map(ref, tracers_out))
        for t in tracers_out:
            t.recipe = eqn
        return tracers_out




def tracers_to_jaxpr(
    tracers_in: List[PartialEvalTracer], tracers_out: List[PartialEvalTracer]
):
    tracer_to_var: Dict[int, Var] = {
        id(t): Var(ArrayShape.like(t.aval)) for t in tracers_in
    }
    constvar_to_val: Dict[int, Any] = {}
    constid_to_var: Dict[int, Var] = {}
    processed_eqns: Set[int] = set()
    eqns: List[JaxprEqn] = []
    for t in toposort(tracers_out, tracer_parents):
        if isinstance(t.recipe, LambdaBindingRecipe):
            assert id(t) in set(list_map(id, tracers_in))
        elif isinstance(t.recipe, ConstRecipe):
            val = t.recipe.val
            var = constid_to_var.get(id(val))
            if var is None:
                aval = ArrayShape.like(Tracer.get_aval(val))
                var = constid_to_var[id(val)] = Var(aval)
                constvar_to_val[var] = val
            tracer_to_var[id(t)] = var
        elif isinstance(t.recipe, JaxprEqnRecipe):
            if id(t.recipe) not in processed_eqns:
                eqns.append(recipe_to_eqn(tracer_to_var, t.recipe))
                processed_eqns.add(id(t.recipe))
        else:
            raise TypeError(t.recipe)

    constvars, constvals = unzip2(constvar_to_val.items())
    in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
    out_vars = [tracer_to_var[id(t)] for t in tracers_out]
    jaxpr = Jaxpr(in_binders, eqns, out_vars)
    typecheck_jaxpr(jaxpr)
    return jaxpr, constvals


def recipe_to_eqn(tracer_to_var: Dict[int, Var], recipe: JaxprEqnRecipe) -> JaxprEqn:
    inputs = [tracer_to_var[id(t)] for t in recipe.tracers_in]
    out_binders = [Var(aval) for aval in recipe.avals_out]
    for t_ref, var in zip(recipe.tracer_refs_out, out_binders):
        if t_ref() is not None:
            tracer_to_var[id(t_ref())] = var
    return JaxprEqn(recipe.prim, inputs, recipe.params, out_binders)


def tracer_parents(t: PartialEvalTracer) -> List[PartialEvalTracer]:
    return t.recipe.tracers_in if isinstance(t.recipe, JaxprEqnRecipe) else []


def toposort(out_nodes: List[Any], parents: Callable[[Any], List[Any]]):
    if not out_nodes:
        return []
    out_nodes = remove_duplicates(out_nodes)

    child_counts = {}
    stack = list(out_nodes)
    while stack:
        node = stack.pop()
        if id(node) in child_counts:
            child_counts[id(node)] += 1
        else:
            child_counts[id(node)] = 1
            stack.extend(parents(node))
    for node in out_nodes:
        child_counts[id(node)] -= 1

    sorted_nodes = []
    childless_nodes = [node for node in out_nodes if not child_counts[id(node)]]
    while childless_nodes:
        node = childless_nodes.pop()
        sorted_nodes.append(node)
        for parent in parents(node):
            if child_counts[id(parent)] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[id(parent)] -= 1

    sorted_nodes = sorted_nodes[::-1]
    check_toposort(sorted_nodes, parents)
    return sorted_nodes


def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if id(x) not in seen and not seen.add(id(x))]


def check_toposort(nodes: List[Any], parents: Callable[[Any], List[Any]]):
    seen = set()
    for node in nodes:
        assert all(id(parent) in seen for parent in parents(node))
        seen.add(id(node))


def vjp_flat(f, *primals_in):
    pvals_in = [PartialVal.known(x) for x in primals_in] + [
        PartialVal.unknown(ArrayShape.like(Tracer.get_aval(x))) for x in primals_in
    ]
    primal_pvals_in, tangent_pvals_in = split_half(pvals_in)

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    jaxpr, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)  # linearize
    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    transpose_inputs = consts + [UndefPrimal(p.aval) for p in tangent_pvals_in]
    f_vjp = lambda *cts: eval_jaxpr_transposed(jaxpr, transpose_inputs, cts)
    return primals_out, f_vjp


def vjp(f, *primals_in):
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, f_vjp_flat = vjp_flat(f, *primals_in_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)

    def f_vjp(*cotangents_out):
        cotangents_out_flat, _ = tree_flatten(cotangents_out)
        cotangents_in_flat = f_vjp_flat(*cotangents_out_flat)
        return tree_unflatten(in_tree, cotangents_in_flat)

    return primals_out, f_vjp


class UndefPrimal(NamedTuple):
    aval: ArrayShape


# NB: the analogous function in JAX is called 'backward_pass'
def eval_jaxpr_transposed(
    jaxpr: Jaxpr, args: List[Any], cotangents: List[Any]
) -> List[Any]:
    primal_env: Dict[Var, Any] = {}
    ct_env: Dict[Var, Any] = {}

    def read_primal(x: Atom) -> Any:
        return primal_env.get(x, UndefPrimal(x.aval)) if type(x) is Var else x.val

    def write_primal(v: Var, val: Any) -> None:
        if type(val) is not UndefPrimal:
            primal_env[v] = val

    def read_cotangent(v: Var) -> Any:
        return ct_env.pop(v, np.zeros(v.aval.shape, v.aval.dtype))

    def write_cotangent(x: Atom, val: Any):
        if type(x) is Var and val is not None:
            ct_env[x] = ct_env[x] + val if x in ct_env else val

    list_map(write_primal, jaxpr.in_binders, args)
    list_map(write_cotangent, jaxpr.outs, cotangents)
    for eqn in jaxpr.eqns[::-1]:
        print(eqn)
        primals_in = list_map(read_primal, eqn.inputs)
        cts_in = list_map(read_cotangent, eqn.out_binders)
        cts_out = eqn.op.T(cts_in, *primals_in, **eqn.params)
        list_map(write_cotangent, eqn.inputs, cts_out)
    ret = [
        read_cotangent(v)
        for v, x in zip(jaxpr.in_binders, args)
        if type(x) is UndefPrimal
    ]
    return ret

def grad(f):
    def gradfun(x, *xs):
        y, f_vjp = vjp(f, x, *xs)
        if np.shape(y) != ():
            raise TypeError
        out = f_vjp(np.ones(np.shape(y), np.result_type(y)))
        return y, out

    return gradfun


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
            lambda d: list_map(tuple, unzip2(sorted(d.items()))),
            lambda keys, vals: dict(zip(keys, vals)),
        )
        self.node_types[UndefPrimal] = NodeType(
            str(UndefPrimal),
            lambda u: (u.aval, ()),
            lambda aval, _: UndefPrimal(aval),
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

# def partial_eval_jaxpr(
#     jaxpr: Jaxpr,
#     in_unknowns: List[bool],
#     instantiate: Optional[List[bool]] = None,
# ) -> Tuple[Jaxpr, Jaxpr, List[bool], int]:
#     env: Dict[Var, bool] = {}
#     residuals: Set[Var] = set()

#     def read(x: Atom) -> bool:
#         return type(x) is Var and env[x]

#     def write(unk: bool, v: Var) -> None:
#         env[v] = unk

#     def new_res(x: Atom) -> Atom:
#         if type(x) is Var:
#             residuals.add(x)
#         return x

#     eqns1, eqns2 = [], []
#     map(write, in_unknowns, jaxpr.in_binders)
#     for eqn in jaxpr.eqns:
#         unks_in = map(read, eqn.inputs)
#         if any(unks_in):
#             inputs = [v if unk else new_res(v) for unk, v in zip(unks_in, eqn.inputs)]
#             eqns2.append(JaxprEqn(eqn.op, inputs, eqn.params, eqn.out_binders))
#             map(partial(write, True), eqn.out_binders)
#         else:
#             eqns1.append(eqn)
#             map(partial(write, False), eqn.out_binders)
#     out_unknowns = map(read, jaxpr.outs)
#     if instantiate is not None:
#         for v, uk, inst in zip(jaxpr.outs, out_unknowns, instantiate):
#             if inst and not uk:
#                 new_res(v)
#         out_unknowns = map(op.or_, out_unknowns, instantiate)

#     residuals, num_res = list(residuals), len(residuals)
#     assert all(type(v) is Var for v in residuals), residuals

#     ins1, ins2 = partition_list(in_unknowns, jaxpr.in_binders)
#     outs1, outs2 = partition_list(out_unknowns, jaxpr.outs)

#     jaxpr1 = Jaxpr(ins1, eqns1, outs1 + residuals)
#     jaxpr2 = Jaxpr(residuals + ins2, eqns2, outs2)
#     typecheck_partial_eval_jaxpr(jaxpr, in_unknowns, out_unknowns, jaxpr1, jaxpr2)

#     return jaxpr1, jaxpr2, out_unknowns, num_res


# def typecheck_partial_eval_jaxpr(jaxpr, unks_in, unks_out, jaxpr1, jaxpr2):
#     jaxpr = typecheck_jaxpr(jaxpr)  # (a1,  a2) -> (b1, b2 )
#     jaxpr1ty = typecheck_jaxpr(jaxpr1)  #  a1       -> (b1, res)
#     jaxpr2ty = typecheck_jaxpr(jaxpr2)  # (res, a2) -> b2

#     a1, a2 = partition_list(unks_in, jaxpr.in_types)
#     b1, b2 = partition_list(unks_out, jaxpr.out_types)
#     b1_, res = split_list(jaxpr1ty.out_types, len(b1))
#     res_, a2_ = split_list(jaxpr2ty.in_types, len(res))
#     b2_ = jaxpr2ty.out_types

#     if jaxpr1ty.in_types != a1:
#         raise TypeError
#     if jaxpr2ty.out_types != b2:
#         raise TypeError
#     if b1 != b1_:
#         raise TypeError
#     if res != res_:
#         raise TypeError
#     if a2 != a2_:
#         raise TypeError
#     if b2 != b2_:
#         raise TypeError


# @lru_cache()
# def transpose_jaxpr(
#     jaxpr: Jaxpr, undef_primals: Tuple[bool, ...]
# ) -> Tuple[Jaxpr, List[Any]]:
#     avals_in, avals_out = typecheck_jaxpr(jaxpr)
#     traceable = partial(eval_jaxpr_transposed, jaxpr)
#     args = [UndefPrimal(a) if u else a for a, u in zip(avals_in, undef_primals)]
#     trans_jaxpr, consts, _ = make_jaxpr(traceable, tuple(args), tuple(avals_out))
#     typecheck_jaxpr(trans_jaxpr)
#     return trans_jaxpr, consts
