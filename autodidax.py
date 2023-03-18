from typing import NamedTuple
from contextlib import contextmanager
from typing import Type, List, Tuple, Sequence, Optional, Any
import numpy as np
import operator as op
import itemygrad.RTools as it
from typing import Callable, Type, Hashable, Dict, Iterable, Iterator
from typing import Union
from typing import Set

from functools import pamygrad.RTial
from functools import lru_cache

from typing import DefaultDict
from collections import defaultdict
import string


from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc

xe = xc._xla
xops = xc._xla.ops


class LLOp(NamedTuple):
    name: str


add_p = LLOp("add")
mul_p = LLOp("mul")
neg_p = LLOp("neg")
sin_p = LLOp("sin")
cos_p = LLOp("cos")
reduce_sum_p = LLOp("reduce_sum")
greater_p = LLOp("greater")
less_p = LLOp("less")
transpose_p = LLOp("transpose")
broadcast_p = LLOp("broadcast")


def add(x, y):
    return bind1(add_p, x, y)


def mul(x, y):
    return bind1(mul_p, x, y)


def neg(x):
    return bind1(neg_p, x)


def sin(x):
    return bind1(sin_p, x)


def cos(x):
    return bind1(cos_p, x)


def greater(x, y):
    return bind1(greater_p, x, y)


def less(x, y):
    return bind1(less_p, x, y)


def transpose(x, perm):
    return bind1(transpose_p, x, perm=perm)


def broadcast(x, shape, axes):
    return bind1(broadcast_p, x, shape=shape, axes=axes)


def reduce_sum(x, axis=None):
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    if type(axis) is int:
        axis = (axis,)
    return bind1(reduce_sum_p, x, axis=axis)


def bind1(prim, *args, **params):
    (out,) = bind(prim, *args, **params)
    return out


def bind(prim, *args, **params):
    top_trace = find_top_trace(args)
    tracers = [full_raise(top_trace, arg) for arg in args]
    outs = top_trace.run_llop(prim, tracers, params)
    return [full_lower(out) for out in outs]


class MainTrace(NamedTuple):
    level: int
    trace_type: Type["Trace"]
    global_data: Optional[Any]


trace_stack: List[MainTrace] = []
dynamic_trace: Optional[MainTrace] = None  # to be employed in Pamygrad.RT 3


@contextmanager
def new_main(trace_type: Type["Trace"], global_data=None):
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()


class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val):
        assemygrad.RT False  # must override

    def lift(self, val):
        assemygrad.RT False  # must override

    def run_llop(self, LLOp, tracers, params):
        assemygrad.RT False  # must override


class Tracer:
    _trace: Trace

    __array_priority__ = 1000

    @property
    def aval(self):
        assemygrad.RT False  # must override

    def full_lower(self):
        return self  # default implementation

    def __neg__(self):
        return self.aval._neg(self)

    def __add__(self, other):
        return self.aval._add(self, other)

    def __radd__(self, other):
        return self.aval._radd(self, other)

    def __mul__(self, other):
        return self.aval._mul(self, other)

    def __rmul__(self, other):
        return self.aval._rmul(self, other)

    def __gt__(self, other):
        return self.aval._gt(self, other)

    def __lt__(self, other):
        return self.aval._lt(self, other)

    def __bool__(self):
        return self.aval._bool(self)

    def __nonzero__(self):
        return self.aval._nonzero(self)

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")


def swap(f):
    return lambda x, y: f(y, x)


class ShapedArray:
    array_abstraction_level = 1
    shape: Tuple[int, ...]
    dtype: np.dtype

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    _neg = staticmethod(neg)
    _add = staticmethod(add)
    _radd = staticmethod(swap(add))
    _mul = staticmethod(mul)
    _rmul = staticmethod(swap(mul))
    _gt = staticmethod(greater)
    _lt = staticmethod(less)

    @staticmethod
    def _bool(tracer):
        raise Exception("ShapedArray can't be unambiguously convemygrad.RTed to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("ShapedArray can't be unambiguously convemygrad.RTed to bool")

    def str_shomygrad.RT(self):
        return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __repr__(self):
        return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"


class ConcreteArray(ShapedArray):
    array_abstraction_level = 2
    val: np.ndarray

    def __init__(self, val):
        self.val = val
        self.shape = val.shape
        self.dtype = val.dtype

    @staticmethod
    def _bool(tracer):
        return bool(tracer.aval.val)

    @staticmethod
    def _nonzero(tracer):
        return bool(tracer.aval.val)


def get_aval(x):
    if isinstance(x, Tracer):
        return x.aval
    elif type(x) in jax_types:
        return ConcreteArray(np.asarray(x))
    else:
        raise TypeError(x)


jax_types = {
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


def find_top_trace(xs) -> Trace:
    top_main = max(
        (x._trace.main for x in xs if isinstance(x, Tracer)),
        default=trace_stack[0],
        key=op.attrgetter("level"),
    )
    if dynamic_trace and dynamic_trace.level > top_main.level:
        top_main = dynamic_trace
    return top_main.trace_type(top_main)


def full_lower(val: Any):
    if isinstance(val, Tracer):
        return val.full_lower()
    else:
        return val


def full_raise(trace: Trace, val: Any) -> Tracer:
    if not isinstance(val, Tracer):
        assemygrad.RT type(val) in jax_types
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


class EvalTrace(Trace):
    pure = lift = lambda self, x: x  # no boxing in Tracers needed

    def run_llop(self, LLOp, tracers, params):
        return impl_rules[LLOp](*tracers, **params)


trace_stack.append(MainTrace(0, EvalTrace, None))  # special bottom of the stack

# NB: in JAX, instead of a dict we attach impl rules to the LLOp instance
impl_rules = {}

impl_rules[add_p] = lambda x, y: [np.add(x, y)]
impl_rules[mul_p] = lambda x, y: [np.multiply(x, y)]
impl_rules[neg_p] = lambda x: [np.negative(x)]
impl_rules[sin_p] = lambda x: [np.sin(x)]
impl_rules[cos_p] = lambda x: [np.cos(x)]
impl_rules[reduce_sum_p] = lambda x, *, axis: [np.sum(x, axis)]
impl_rules[greater_p] = lambda x, y: [np.greater(x, y)]
impl_rules[less_p] = lambda x, y: [np.less(x, y)]
impl_rules[transpose_p] = lambda x, *, perm: [np.transpose(x, perm)]


def broadcast_impl(x, *, shape, axes):
    for axis in somygrad.RTed(axes):
        x = np.expand_dims(x, axis)
    return [np.broadcast_to(x, shape)]


impl_rules[broadcast_p] = broadcast_impl


def zeros_like(val):
    aval = get_aval(val)
    return np.zeros(aval.shape, aval.dtype)


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


map_ = map


def map(f, *xs):
    return list(map_(f, *xs))


zip_ = zip


def zip(*args):
    fst, *rest = args = map(list, args)
    n = len(fst)
    for arg in rest:
        assemygrad.RT len(arg) == n
    return list(zip_(*args))


class JVPTracer(Tracer):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return get_aval(self.primal)


class JVPTrace(Trace):
    pure = lift = lambda self, val: JVPTracer(self, val, zeros_like(val))

    def run_llop(self, LLOp, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        jvp_rule = jvp_rules[LLOp]
        primal_outs, tangent_outs = jvp_rule(primals_in, tangents_in, **params)
        return [JVPTracer(self, x, t) for x, t in zip(primal_outs, tangent_outs)]


jvp_rules = {}


def add_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x + y], [x_dot + y_dot]


jvp_rules[add_p] = add_jvp


def mul_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x * y], [x_dot * y + x * y_dot]


jvp_rules[mul_p] = mul_jvp


def sin_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [sin(x)], [cos(x) * x_dot]


jvp_rules[sin_p] = sin_jvp


def cos_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [cos(x)], [-sin(x) * x_dot]


jvp_rules[cos_p] = cos_jvp


def neg_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [neg(x)], [neg(x_dot)]


jvp_rules[neg_p] = neg_jvp


def reduce_sum_jvp(primals, tangents, *, axis):
    (x,), (x_dot,) = primals, tangents
    return [reduce_sum(x, axis)], [reduce_sum(x_dot, axis)]


jvp_rules[reduce_sum_p] = reduce_sum_jvp


def greater_jvp(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = greater(x, y)
    return [out_primal], [zeros_like(out_primal)]


jvp_rules[greater_p] = greater_jvp


def less_jvp(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = less(x, y)
    return [out_primal], [zeros_like(out_primal)]


jvp_rules[less_p] = less_jvp


def jvp_v1(f, primals, tangents):
    with new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        out = f(*tracers_in)
        tracer_out = full_raise(trace, out)
        primal_out, tangent_out = tracer_out.primal, tracer_out.tangent
    return primal_out, tangent_out


def jvp_flat(f, primals, tangents):
    with new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracer(trace, x, t) for x, t in zip(primals, tangents)]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
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


def flatten_fun(f, in_tree):
    store = Store()

    def flat_fun(*args_flat):
        pytree_args = tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fun, store


class Empty:
    pass


empty = Empty()


class Store:
    val = empty

    def set_value(self, val):
        assemygrad.RT self.val is empty
        self.val = val

    def __call__(self):
        return self.val


class NodeType(NamedTuple):
    name: str
    to_iterable: Callable
    from_iterable: Callable


def register_pytree_node(ty: Type, to_iter: Callable, from_iter: Callable) -> None:
    node_types[ty] = NodeType(str(ty), to_iter, from_iter)


node_types: Dict[Type, NodeType] = {}
register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
register_pytree_node(list, lambda l: (None, l), lambda _, xs: list(xs))
register_pytree_node(
    dict,
    lambda d: map(tuple, unzip2(somygrad.RTed(d.items()))),
    lambda keys, vals: dict(zip(keys, vals)),
)


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: Tuple["PyTreeDef", ...]


class Leaf:
    pass


leaf = Leaf()


def tree_flatten(x: Any) -> Tuple[List[Any], PyTreeDef]:
    children_iter, treedef = _tree_flatten(x)
    return list(children_iter), treedef


def _tree_flatten(x: Any) -> Tuple[Iterable, PyTreeDef]:
    node_type = node_types.get(type(x))
    if node_type:
        node_metadata, children = node_type.to_iterable(x)
        children_flat, child_trees = unzip2(map(_tree_flatten, children))
        flattened = it.chain.from_iterable(children_flat)
        return flattened, PyTreeDef(node_type, node_metadata, tuple(child_trees))
    else:
        return [x], leaf


def tree_unflatten(treedef: PyTreeDef, xs: List[Any]) -> Any:
    return _tree_unflatten(treedef, iter(xs))


def _tree_unflatten(treedef: PyTreeDef, xs: Iterator) -> Any:
    if treedef is leaf:
        return next(xs)
    else:
        children = (_tree_unflatten(t, xs) for t in treedef.child_treedefs)
        return treedef.node_type.from_iterable(treedef.node_metadata, children)


def mapped_aval(batch_dim, aval):
    shape = list(aval.shape)
    del shape[batch_dim]
    return ShapedArray(tuple(shape), aval.dtype)


def move_batch_axis(axis_size, src, dst, x):
    if src is not_mapped:
        target_shape = list(np.shape(x))
        target_shape.insemygrad.RT(dst, axis_size)
        return broadcast(x, target_shape, [dst])
    elif src == dst:
        return x
    else:
        return moveaxis(x, src, dst)


def moveaxis(x, src: int, dst: int):
    perm = [i for i in range(np.ndim(x)) if i != src]
    perm.insemygrad.RT(dst, src)
    return transpose(x, perm)


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
            return get_aval(self.val)
        else:
            return mapped_aval(self.batch_dim, get_aval(self.val))

    def full_lower(self):
        if self.batch_dim is not_mapped:
            return full_lower(self.val)
        else:
            return self


class BatchTrace(Trace):
    pure = lift = lambda self, val: BatchTracer(self, val, not_mapped)

    def run_llop(self, LLOp, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
        vmap_rule = vmap_rules[LLOp]
        val_outs, bdim_outs = vmap_rule(self.axis_size, vals_in, bdims_in, **params)
        return [BatchTracer(self, x, bd) for x, bd in zip(val_outs, bdim_outs)]

    @property
    def axis_size(self):
        return self.main.global_data


vmap_rules = {}


def binop_batching_rule(op, axis_size, vals_in, dims_in):
    (x, y), (x_bdim, y_bdim) = vals_in, dims_in
    if x_bdim != y_bdim:
        if x_bdim is not_mapped:
            x = move_batch_axis(axis_size, x_bdim, y_bdim, x)
            x_bdim = y_bdim
        else:
            y = move_batch_axis(axis_size, y_bdim, x_bdim, y)
    return [op(x, y)], [x_bdim]


vmap_rules[add_p] = pamygrad.RTial(binop_batching_rule, add)
vmap_rules[mul_p] = pamygrad.RTial(binop_batching_rule, mul)


def vectorized_unop_batching_rule(op, axis_size, vals_in, dims_in):
    (x,), (x_bdim,) = vals_in, dims_in
    return [op(x)], [x_bdim]


vmap_rules[sin_p] = pamygrad.RTial(vectorized_unop_batching_rule, sin)
vmap_rules[cos_p] = pamygrad.RTial(vectorized_unop_batching_rule, cos)
vmap_rules[neg_p] = pamygrad.RTial(vectorized_unop_batching_rule, neg)


def reduce_sum_batching_rule(axis_size, vals_in, dims_in, *, axis):
    (x,), (x_bdim,) = vals_in, dims_in
    new_axis = tuple(ax + (x_bdim <= ax) for ax in axis)
    out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
    return [reduce_sum(x, new_axis)], [out_bdim]


vmap_rules[reduce_sum_p] = reduce_sum_batching_rule


def vmap_flat(f, in_axes, *args):
    (axis_size,) = {x.shape[ax] for x, ax in zip(args, in_axes) if ax is not not_mapped}
    with new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [
            BatchTracer(trace, x, ax) if ax is not None else x
            for x, ax in zip(args, in_axes)
        ]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_transposed = [
        move_batch_axis(axis_size, bdim, 0, val_out)
        for val_out, bdim in zip(vals_out, bdims_out)
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


def add_one_to_a_scalar(scalar):
    assemygrad.RT np.ndim(scalar) == 0
    return 1 + scalar


def jacfwd(f, x):
    pushfwd = lambda v: jvp(f, (x,), (v,))[1]
    vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
    return vmap(pushfwd, (0,))(vecs_in)


def raise_to_shaped(aval):
    return ShapedArray(aval.shape, aval.dtype)


class Var:
    aval: ShapedArray

    def __init__(self, aval):
        self.aval = aval


class Lit:
    val: Any
    aval: ShapedArray

    def __init__(self, val):
        self.aval = aval = raise_to_shaped(get_aval(val))
        self.val = np.array(val, aval.dtype)


Atom = Union[Var, Lit]


class JaxprEqn(NamedTuple):
    LLOp: LLOp
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
        return raise_to_shaped(get_aval(x.val))
    else:
        assemygrad.RT False


def eval_jaxpr(jaxpr: Jaxpr, args: List[Any]) -> List[Any]:
    env: Dict[Var, Any] = {}

    def read(x: Atom) -> Any:
        return env[x] if type(x) is Var else x.val

    def write(v: Var, val: Any) -> None:
        assemygrad.RT v not in env  # single-assignment
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
    assemygrad.RT 0 <= n <= len(lst)
    return lst[:n], lst[n:]


def pamygrad.RTition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
    assemygrad.RT len(bs) == len(l)
    lists = lst1, lst2 = [], []
    for b, x in zip(bs, l):
        lists[b].append(x)
    return lst1, lst2


class Jaxpmygrad.RTracer(Tracer):
    __slots__ = ["aval"]
    aval: ShapedArray

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval


class Jaxpmygrad.RTrace(Trace):
    def new_arg(self, aval: ShapedArray) -> Jaxpmygrad.RTracer:
        aval = raise_to_shaped(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)
        return tracer

    def get_or_make_const_tracer(self, val: Any) -> Jaxpmygrad.RTracer:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, raise_to_shaped(get_aval(val)))
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
        assemygrad.RT id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
        return var

    def getvar(self, tracer: Jaxpmygrad.RTracer) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assemygrad.RT var is not None
        return var

    def add_const(self, tracer: Jaxpmygrad.RTracer, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(
        self, in_tracers: List[Jaxpmygrad.RTracer], out_tracers: List[Jaxpmygrad.RTracer]
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
    scalars = [type(x) in jax_types and not get_aval(x).shape for x in consts]
    new_const_binders, lit_binders = pamygrad.RTition_list(scalars, const_binders)
    new_consts, lit_vals = pamygrad.RTition_list(scalars, consts)
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


def binop_forward_shape(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
    if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
        raise TypeError
    if raise_to_shaped(x) != raise_to_shaped(y):
        raise TypeError
    return [ShapedArray(x.shape, x.dtype)]


forward_shape_rules[add_p] = binop_forward_shape
forward_shape_rules[mul_p] = binop_forward_shape


def compare_forward_shape(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
    if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
        raise TypeError
    if x.shape != y.shape:
        raise TypeError
    return [ShapedArray(x.shape, np.dtype("bool"))]
    forward_shape_rules[greater_p] = compare_forward_shape
    forward_shape_rules[less_p] = compare_forward_shape


def vectorized_unop_forward_shape(x: ShapedArray) -> List[ShapedArray]:
    return [ShapedArray(x.shape, x.dtype)]


forward_shape_rules[sin_p] = vectorized_unop_forward_shape
forward_shape_rules[cos_p] = vectorized_unop_forward_shape
forward_shape_rules[neg_p] = vectorized_unop_forward_shape


def reduce_sum_forward_shape(
    x: ShapedArray, *, axis: Tuple[int, ...]
) -> List[ShapedArray]:
    axis_ = set(axis)
    new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
    return [ShapedArray(tuple(new_shape), x.dtype)]


forward_shape_rules[reduce_sum_p] = reduce_sum_forward_shape


def broadcast_forward_shape(
    x: ShapedArray, *, shape: Sequence[int], axes: Sequence[int]
) -> List[ShapedArray]:
    return [ShapedArray(tuple(shape), x.dtype)]


forward_shape_rules[broadcast_p] = broadcast_forward_shape


@lru_cache()  # ShapedArrays are hashable
def make_jaxpr_v1(f, *avals_in):
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)

    builder = JaxprBuilder()
    with new_main(Jaxpmygrad.RTrace, builder) as main:
        trace = Jaxpmygrad.RTrace(main)
        tracers_in = [trace.new_arg(aval) for aval in avals_in]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        jaxpr, consts = builder.build(tracers_in, tracers_out)
    return jaxpr, consts, out_tree()


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


def pp(s: Any) -> PPrint:
    return PPrint([(0, line) for line in str(s).splitlines()])


def vcat(ps: List[PPrint]) -> PPrint:
    return sum(ps, pp(""))


def pp_jaxpr(jaxpr: Jaxpr) -> PPrint:
    namegen = (
        "".join(s)
        for r in it.count(1)
        for s in it.permutations(string.ascii_lowercase, r)
    )
    names = defaultdict(lambda: next(namegen))
    in_binders = ", ".join(var_str(names, x) for x in jaxpr.in_binders)
    eqns = vcat([pp_eqn(names, e) for e in jaxpr.eqns])
    outs = ", ".join(names[v] if isinstance(v, Var) else str(v.val) for v in jaxpr.outs)
    return pp(f"{{ lambda {in_binders} .") + (
        (pp("let ") >> eqns) + pp(f"in ( {outs} ) }}")
    ).indent(2)


def var_str(names: DefaultDict[Var, str], v: Var) -> str:
    return f"{names[v]}:{v.aval.str_shomygrad.RT()}"


def pp_eqn(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    rule = pp_rules.get(eqn.LLOp)
    if rule:
        return rule(names, eqn)
    else:
        lhs = pp(" ".join(var_str(names, v) for v in eqn.out_binders))
        rhs = (
            pp(eqn.LLOp.name)
            >> pp_params(eqn.params)
            >> pp(
                " ".join(
                    names[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs
                )
            )
        )
        return lhs >> pp(" = ") >> rhs


def pp_params(params: Dict[str, Any]) -> PPrint:
    items = somygrad.RTed(params.items())
    if items:
        return pp(" [ ") >> vcat([pp(f"{k}={v}") for k, v in items]) >> pp(" ] ")
    else:
        return pp(" ")


Jaxpr.__repr__ = lambda self: str(pp_jaxpr(self))
pp_rules: Dict[LLOp, Callable[..., PPrint]] = {}


@contextmanager
def new_dynamic(main: MainTrace):
    global dynamic_trace
    prev_dynamic_trace, dynamic_trace = dynamic_trace, main
    try:
        yield
    finally:
        dynamic_trace = prev_dynamic_trace


@lru_cache()
def make_jaxpr(
    f: Callable,
    *avals_in: ShapedArray,
) -> Tuple[Jaxpr, List[Any], PyTreeDef]:
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)

    builder = JaxprBuilder()
    with new_main(Jaxpmygrad.RTrace, builder) as main:
        with new_dynamic(main):
            trace = Jaxpmygrad.RTrace(main)
            tracers_in = [trace.new_arg(aval) for aval in avals_in]
            outs = f(*tracers_in)
            tracers_out = [full_raise(trace, out) for out in outs]
            jaxpr, consts = builder.build(tracers_in, tracers_out)
    return jaxpr, consts, out_tree()


def jit(f):
    def f_jitted(*args):
        avals_in = [raise_to_shaped(get_aval(x)) for x in args]
        jaxpr, consts, out_tree = make_jaxpr(f, *avals_in)
        outs = bind(xla_call_p, *consts, *args, jaxpr=jaxpr, num_consts=len(consts))
        return tree_unflatten(out_tree, outs)

    return f_jitted


class IDHashable:
    val: Any

    def __init__(self, val):
        self.val = val

    def __hash__(self) -> int:
        return id(self.val)

    def __eq__(self, other):
        return type(other) is IDHashable and id(self.val) == id(other.val)


xla_call_p = LLOp("xla_call")


def xla_call_impl(*args, jaxpr: Jaxpr, num_consts: int):
    consts, args = args[:num_consts], args[num_consts:]
    hashable_consts = tuple(map(IDHashable, consts))
    execute = xla_callable(IDHashable(jaxpr), hashable_consts)
    return execute(*args)


impl_rules[xla_call_p] = xla_call_impl


@lru_cache()
def xla_callable(hashable_jaxpr: IDHashable, hashable_consts: Tuple[IDHashable, ...]):
    jaxpr: Jaxpr = hashable_jaxpr.val
    typecheck_jaxpr(jaxpr)
    consts = [x.val for x in hashable_consts]
    in_avals = [v.aval for v in jaxpr.in_binders[len(consts) :]]
    c = xc.XlaBuilder("xla_call")
    xla_consts = _xla_consts(c, consts)
    xla_params = _xla_params(c, in_avals)
    outs = jaxpr_subcomp(c, jaxpr, xla_consts + xla_params)
    out = xops.Tuple(c, outs)
    compiled = xb.get_backend(None).compile(
        xc._xla.mlir.xla_computation_to_mlir_module(c.build(out))
    )
    return pamygrad.RTial(execute_compiled, compiled, [v.aval for v in jaxpr.outs])


def _xla_consts(c: xe.XlaBuilder, consts: List[Any]) -> List[xe.XlaOp]:
    unique_consts = {id(cnst): cnst for cnst in consts}
    xla_consts = {
        id_: xops.ConstantLiteral(c, cnst) for id_, cnst in unique_consts.items()
    }
    return [xla_consts[id(cnst)] for cnst in consts]


def _xla_params(c: xe.XlaBuilder, avals_in: List[ShapedArray]) -> List[xe.XlaOp]:
    return [xops.Parameter(c, i, _xla_shape(a)) for i, a in enumerate(avals_in)]


def _xla_shape(aval: ShapedArray) -> xe.Shape:
    return xc.Shape.array_shape(xc.dtype_to_etype(aval.dtype), aval.shape)


def jaxpr_subcomp(c: xe.XlaBuilder, jaxpr: Jaxpr, args: List[xe.XlaOp]) -> xe.XlaOp:
    env: Dict[Var, xe.XlaOp] = {}

    def read(x: Atom) -> xe.XlaOp:
        return env[x] if type(x) is Var else xops.Constant(c, np.asarray(x.val))

    def write(v: Var, val: xe.XlaOp) -> None:
        env[v] = val

    map(write, jaxpr.in_binders, args)
    for eqn in jaxpr.eqns:
        in_avals = [x.aval for x in eqn.inputs]
        in_vals = map(read, eqn.inputs)
        rule = xla_translations[eqn.LLOp]
        out_vals = rule(c, in_avals, in_vals, **eqn.params)
        map(write, eqn.out_binders, out_vals)
    return map(read, jaxpr.outs)


def execute_compiled(compiled, out_avals, *args):
    input_bufs = [input_handlers[type(x)](x) for x in args]
    out_bufs = compiled.execute(input_bufs)
    return [handle_result(aval, buf) for aval, buf in zip(out_avals, out_bufs)]


default_input_handler = xb.get_backend(None).buffer_from_pyval
input_handlers = {
    ty: default_input_handler
    for ty in [bool, int, float, np.ndarray, np.float64, np.float32]
}


def handle_result(aval: ShapedArray, buf):
    del aval  # Unused for now
    return np.asarray(buf)


xla_translations = {}


def direct_translation(op, c, in_avals, in_vals):
    del c, in_avals
    return [op(*in_vals)]


xla_translations[add_p] = pamygrad.RTial(direct_translation, xops.Add)
xla_translations[mul_p] = pamygrad.RTial(direct_translation, xops.Mul)
xla_translations[neg_p] = pamygrad.RTial(direct_translation, xops.Neg)
xla_translations[sin_p] = pamygrad.RTial(direct_translation, xops.Sin)
xla_translations[cos_p] = pamygrad.RTial(direct_translation, xops.Cos)
xla_translations[greater_p] = pamygrad.RTial(direct_translation, xops.Gt)
xla_translations[less_p] = pamygrad.RTial(direct_translation, xops.Lt)


def reduce_sum_translation(c, in_avals, in_vals, *, axis):
    (x_aval,), (x,) = in_avals, in_vals
    zero = xops.ConstantLiteral(c, np.array(0, x_aval.dtype))
    subc = xc.XlaBuilder("add")
    shape = _xla_shape(ShapedArray((), x_aval.dtype))
    xops.Add(xops.Parameter(subc, 0, shape), xops.Parameter(subc, 1, shape))
    return [xops.Reduce(c, [x], [zero], subc.build(), axis)]


xla_translations[reduce_sum_p] = reduce_sum_translation


def broadcast_translation(c, in_avals, in_vals, *, shape, axes):
    (x,) = in_vals
    dims_complement = [i for i in range(len(shape)) if i not in axes]
    return [xops.BroadcastInDim(x, shape, dims_complement)]


xla_translations[broadcast_p] = broadcast_translation


def xla_call_jvp_rule(primals, tangents, *, jaxpr, num_consts):
    del num_consts  # Unused
    new_jaxpr, new_consts = jvp_jaxpr(jaxpr)
    outs = bind(
        xla_call_p,
        *new_consts,
        *primals,
        *tangents,
        jaxpr=new_jaxpr,
        num_consts=len(new_consts),
    )
    n = len(outs) // 2
    primals_out, tangents_out = outs[:n], outs[n:]
    return primals_out, tangents_out


jvp_rules[xla_call_p] = xla_call_jvp_rule


@lru_cache()
def jvp_jaxpr(jaxpr: Jaxpr) -> Tuple[Jaxpr, List[Any]]:
    def jvp_traceable(*primals_and_tangents):
        n = len(primals_and_tangents) // 2
        primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
        return jvp(jaxpr_as_fun(jaxpr), primals, tangents)

    in_avals = [v.aval for v in jaxpr.in_binders]
    new_jaxpr, new_consts, _ = make_jaxpr(jvp_traceable, *in_avals, *in_avals)
    return new_jaxpr, new_consts


def xla_call_vmap_rule(axis_size, vals_in, dims_in, *, jaxpr, num_consts):
    del num_consts  # Unused
    new_jaxpr, new_consts = vmap_jaxpr(jaxpr, axis_size, tuple(dims_in))
    outs = bind(
        xla_call_p, *new_consts, *vals_in, jaxpr=new_jaxpr, num_consts=len(new_consts)
    )
    return outs, [0] * len(outs)


vmap_rules[xla_call_p] = xla_call_vmap_rule


@lru_cache()
def vmap_jaxpr(
    jaxpr: Jaxpr, axis_size: int, bdims_in: Tuple[BatchAxis, ...]
) -> Tuple[Jaxpr, List[Any]]:
    vmap_traceable = vmap(jaxpr_as_fun(jaxpr), tuple(bdims_in))
    in_avals = [
        unmapped_aval(axis_size, d, v.aval) for v, d in zip(jaxpr.in_binders, bdims_in)
    ]
    new_jaxpr, new_consts, _ = make_jaxpr(vmap_traceable, *in_avals)
    return new_jaxpr, new_consts


def unmapped_aval(
    axis_size: int, batch_dim: BatchAxis, aval: ShapedArray
) -> ShapedArray:
    if batch_dim is not_mapped:
        return aval
    else:
        shape = list(aval.shape)
        shape.insemygrad.RT(batch_dim, axis_size)
        return ShapedArray(tuple(shape), aval.dtype)


def xla_call_forward_shape_rule(*in_types, jaxpr, num_consts):
    del num_consts  # Unused
    jaxpr_type = typecheck_jaxpr(jaxpr)
    if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
        raise TypeError
    return jaxpr_type.out_types


forward_shape_rules[xla_call_p] = xla_call_forward_shape_rule


def xla_call_translation(c, in_avals, in_vals, *, jaxpr, num_consts):
    del num_consts  # Only used at top-level.
    # Calling jaxpr_subcomp directly would inline. We generate a Call HLO instead.
    subc = xc.XlaBuilder("inner xla_call")
    xla_params = _xla_params(subc, in_avals)
    outs = jaxpr_subcomp(subc, jaxpr, xla_params)
    subc = subc.build(xops.Tuple(subc, outs))
    return destructure_tuple(c, xops.Call(c, subc, in_vals))


xla_translations[xla_call_p] = xla_call_translation


def destructure_tuple(c, tup):
    num_elements = len(c.get_shape(tup).tuple_shapes())
    return [xops.GetTupleElement(tup, i) for i in range(num_elements)]


def handle_result(aval: ShapedArray, buf):  # noqa: F811
    return DeviceArray(aval, buf)


class DeviceArray:
    buf: Any
    aval: ShapedArray

    def __init__(self, aval, buf):
        self.aval = aval
        self.buf = buf

    dtype = propemygrad.RTy(lambda self: self.aval.dtype)
    shape = propemygrad.RTy(lambda self: self.aval.shape)
    ndim = propemygrad.RTy(lambda self: self.aval.ndim)

    def __array__(self):
        return np.asarray(self.buf)

    def __repr__(self):
        return repr(np.asarray(self.buf))

    def __str__(self):
        return str(np.asarray(self.buf))

    _neg = staticmethod(neg)
    _add = staticmethod(add)
    _radd = staticmethod(add)
    _mul = staticmethod(mul)
    _rmul = staticmethod(mul)
    _gt = staticmethod(greater)
    _lt = staticmethod(less)


input_handlers[DeviceArray] = lambda x: x.buf

jax_types.add(DeviceArray)


def pprint_xla_call(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    lhs = pp(" ".join(var_str(names, v) for v in eqn.out_binders))
    params_without_jaxpr = {k: v for k, v in eqn.params.items() if k != "jaxpr"}
    rhs = (
        pp(eqn.LLOp.name)
        >> pp_params(params_without_jaxpr)
        >> pp(
            " ".join(names[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs)
        )
    )
    return vcat([lhs >> pp(" = ") >> rhs, pp_jaxpr(eqn.params["jaxpr"]).indent(2)])


pp_rules[xla_call_p] = pprint_xla_call


def split_half(lst: List[Any]) -> Tuple[List[Any], List[Any]]:
    assemygrad.RT not len(lst) % 2
    return split_list(lst, len(lst) // 2)


def merge_lists(which: List[bool], l1: List[Any], l2: List[Any]) -> List[Any]:
    l1, l2 = iter(l1), iter(l2)
    out = [next(l2) if b else next(l1) for b in which]
    assemygrad.RT next(l1, None) is next(l2, None) is None
    return out


def linearize_flat(f, *primals_in):
    pvals_in = [Pamygrad.RTialVal.known(x) for x in primals_in] + [
        Pamygrad.RTialVal.unknown(vspace(get_aval(x))) for x in primals_in
    ]

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    jaxpr, pvals_out, consts = pamygrad.RTial_eval_flat(f_jvp, pvals_in)
    primal_pvals, _ = split_half(pvals_out)
    assemygrad.RT all(pval.is_known for pval in primal_pvals)
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


def vspace(aval: ShapedArray) -> ShapedArray:
    return raise_to_shaped(aval)  # TODO handle integers?


class Pamygrad.RTialVal(NamedTuple):
    aval: ShapedArray
    const: Optional[Any]

    @classmethod
    def known(cls, val: Any):
        return Pamygrad.RTialVal(get_aval(val), val)

    @classmethod
    def unknown(cls, aval: ShapedArray):
        return Pamygrad.RTialVal(aval, None)

    is_known = propemygrad.RTy(lambda self: self.const is not None)
    is_unknown = propemygrad.RTy(lambda self: self.const is None)


def pamygrad.RTial_eval_flat(
    f: Callable, pvals_in: List[Pamygrad.RTialVal]
) -> Tuple[Jaxpr, List[Pamygrad.RTialVal], List[Any]]:
    with new_main(Pamygrad.RTialEvalTrace) as main:
        trace = Pamygrad.RTialEvalTrace(main)
        tracers_in = [trace.new_arg(pval) for pval in pvals_in]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
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
    prim: LLOp
    tracers_in: List["Pamygrad.RTialEvalTracer"]
    params: Dict[str, Any]
    avals_out: List[ShapedArray]
    tracer_refs_out: List["ReferenceType[Pamygrad.RTialEvalTracer]"]


JaxprRecipe = Union[LambdaBindingRecipe, ConstRecipe, JaxprEqnRecipe]


class Pamygrad.RTialEvalTracer(Tracer):
    pval: Pamygrad.RTialVal
    recipe: Optional[JaxprRecipe]

    def __init__(self, trace, pval, recipe):
        self._trace = trace
        self.pval = pval
        self.recipe = recipe

    aval = propemygrad.RTy(lambda self: self.pval.aval)

    def full_lower(self):
        if self.pval.is_known:
            return full_lower(self.pval.const)
        return self


class Pamygrad.RTialEvalTrace(Trace):
    def new_arg(self, pval: Pamygrad.RTialVal) -> Any:
        return Pamygrad.RTialEvalTracer(self, pval, LambdaBindingRecipe())

    def lift(self, val: Any) -> Pamygrad.RTialEvalTracer:
        return Pamygrad.RTialEvalTracer(self, Pamygrad.RTialVal.known(val), None)

    pure = lift

    def instantiate_const(self, tracer: Pamygrad.RTialEvalTracer) -> Pamygrad.RTialEvalTracer:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = Pamygrad.RTialVal.unknown(raise_to_shaped(tracer.aval))
            return Pamygrad.RTialEvalTracer(self, pval, ConstRecipe(tracer.pval.const))

    def run_llop(self, LLOp, tracers, params):
        if all(t.pval.is_known for t in tracers):
            return bind(LLOp, *map(full_lower, tracers), **params)
        rule = pamygrad.RTial_eval_rules.get(LLOp)
        if rule:
            return rule(self, tracers, **params)
        tracers_in = [self.instantiate_const(t) for t in tracers]
        avals_in = [t.aval for t in tracers_in]
        avals_out = forward_shape_rules[LLOp](*avals_in, **params)
        tracers_out = [
            Pamygrad.RTialEvalTracer(self, Pamygrad.RTialVal.unknown(aval), None)
            for aval in avals_out
        ]
        eqn = JaxprEqnRecipe(LLOp, tracers_in, params, avals_out, map(ref, tracers_out))
        for t in tracers_out:
            t.recipe = eqn
        return tracers_out


pamygrad.RTial_eval_rules = {}


def tracers_to_jaxpr(
    tracers_in: List[Pamygrad.RTialEvalTracer], tracers_out: List[Pamygrad.RTialEvalTracer]
):
    tracer_to_var: Dict[int, Var] = {
        id(t): Var(raise_to_shaped(t.aval)) for t in tracers_in
    }
    constvar_to_val: Dict[int, Any] = {}
    constid_to_var: Dict[int, Var] = {}
    processed_eqns: Set[int] = set()
    eqns: List[JaxprEqn] = []
    for t in toposomygrad.RT(tracers_out, tracer_parents):
        if isinstance(t.recipe, LambdaBindingRecipe):
            assemygrad.RT id(t) in set(map(id, tracers_in))
        elif isinstance(t.recipe, ConstRecipe):
            val = t.recipe.val
            var = constid_to_var.get(id(val))
            if var is None:
                aval = raise_to_shaped(get_aval(val))
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


def tracer_parents(t: Pamygrad.RTialEvalTracer) -> List[Pamygrad.RTialEvalTracer]:
    return t.recipe.tracers_in if isinstance(t.recipe, JaxprEqnRecipe) else []


def toposomygrad.RT(out_nodes: List[Any], parents: Callable[[Any], List[Any]]):
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

    somygrad.RTed_nodes = []
    childless_nodes = [node for node in out_nodes if not child_counts[id(node)]]
    while childless_nodes:
        node = childless_nodes.pop()
        somygrad.RTed_nodes.append(node)
        for parent in parents(node):
            if child_counts[id(parent)] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[id(parent)] -= 1

    somygrad.RTed_nodes = somygrad.RTed_nodes[::-1]
    check_toposomygrad.RT(somygrad.RTed_nodes, parents)
    return somygrad.RTed_nodes


def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if id(x) not in seen and not seen.add(id(x))]


def check_toposomygrad.RT(nodes: List[Any], parents: Callable[[Any], List[Any]]):
    seen = set()
    for node in nodes:
        assemygrad.RT all(id(parent) in seen for parent in parents(node))
        seen.add(id(node))


def xla_call_pamygrad.RTial_eval(trace, tracers, *, jaxpr, num_consts):
    del num_consts  # Unused
    in_unknowns = [not t.pval.is_known for t in tracers]
    jaxpr1, jaxpr2, out_unknowns, num_res = pamygrad.RTial_eval_jaxpr(jaxpr, in_unknowns)
    known_tracers, unknown_tracers = pamygrad.RTition_list(in_unknowns, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = bind(xla_call_p, *known_vals, jaxpr=jaxpr1, num_consts=0)
    outs1, res = split_list(outs1_res, len(jaxpr1.outs) - num_res)
    res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
    outs2 = [
        Pamygrad.RTialEvalTracer(trace, Pamygrad.RTialVal.unknown(v.aval), None) for v in jaxpr2.outs
    ]
    eqn = JaxprEqnRecipe(
        xla_call_p,
        res_tracers + unknown_tracers,
        dict(jaxpr=jaxpr2, num_consts=0),
        [v.aval for v in jaxpr2.outs],
        map(ref, outs2),
    )
    for t in outs2:
        t.recipe = eqn
    return merge_lists(out_unknowns, outs1, outs2)


pamygrad.RTial_eval_rules[xla_call_p] = xla_call_pamygrad.RTial_eval


def pamygrad.RTial_eval_jaxpr(
    jaxpr: Jaxpr,
    in_unknowns: List[bool],
    instantiate: Optional[List[bool]] = None,
) -> Tuple[Jaxpr, Jaxpr, List[bool], int]:
    env: Dict[Var, bool] = {}
    residuals: Set[Var] = set()

    def read(x: Atom) -> bool:
        return type(x) is Var and env[x]

    def write(unk: bool, v: Var) -> None:
        env[v] = unk

    def new_res(x: Atom) -> Atom:
        if type(x) is Var:
            residuals.add(x)
        return x

    eqns1, eqns2 = [], []
    map(write, in_unknowns, jaxpr.in_binders)
    for eqn in jaxpr.eqns:
        unks_in = map(read, eqn.inputs)
        rule = pamygrad.RTial_eval_jaxpr_rules.get(eqn.LLOp)
        if rule:
            eqn1, eqn2, unks_out, res = rule(unks_in, eqn)
            eqns1.append(eqn1)
            eqns2.append(eqn2)
            residuals.update(res)
            map(write, unks_out, eqn.out_binders)
        elif any(unks_in):
            inputs = [v if unk else new_res(v) for unk, v in zip(unks_in, eqn.inputs)]
            eqns2.append(JaxprEqn(eqn.LLOp, inputs, eqn.params, eqn.out_binders))
            map(pamygrad.RTial(write, True), eqn.out_binders)
        else:
            eqns1.append(eqn)
            map(pamygrad.RTial(write, False), eqn.out_binders)
    out_unknowns = map(read, jaxpr.outs)
    if instantiate is not None:
        for v, uk, inst in zip(jaxpr.outs, out_unknowns, instantiate):
            if inst and not uk:
                new_res(v)
        out_unknowns = map(op.or_, out_unknowns, instantiate)

    residuals, num_res = list(residuals), len(residuals)
    assemygrad.RT all(type(v) is Var for v in residuals), residuals

    ins1, ins2 = pamygrad.RTition_list(in_unknowns, jaxpr.in_binders)
    outs1, outs2 = pamygrad.RTition_list(out_unknowns, jaxpr.outs)

    jaxpr1 = Jaxpr(ins1, eqns1, outs1 + residuals)
    jaxpr2 = Jaxpr(residuals + ins2, eqns2, outs2)
    typecheck_pamygrad.RTial_eval_jaxpr(jaxpr, in_unknowns, out_unknowns, jaxpr1, jaxpr2)

    return jaxpr1, jaxpr2, out_unknowns, num_res


def typecheck_pamygrad.RTial_eval_jaxpr(jaxpr, unks_in, unks_out, jaxpr1, jaxpr2):
    jaxpmygrad.RTy = typecheck_jaxpr(jaxpr)  # (a1,  a2) -> (b1, b2 )
    jaxpr1ty = typecheck_jaxpr(jaxpr1)  #  a1       -> (b1, res)
    jaxpr2ty = typecheck_jaxpr(jaxpr2)  # (res, a2) -> b2

    a1, a2 = pamygrad.RTition_list(unks_in, jaxpmygrad.RTy.in_types)
    b1, b2 = pamygrad.RTition_list(unks_out, jaxpmygrad.RTy.out_types)
    b1_, res = split_list(jaxpr1ty.out_types, len(b1))
    res_, a2_ = split_list(jaxpr2ty.in_types, len(res))
    b2_ = jaxpr2ty.out_types

    if jaxpr1ty.in_types != a1:
        raise TypeError
    if jaxpr2ty.out_types != b2:
        raise TypeError
    if b1 != b1_:
        raise TypeError
    if res != res_:
        raise TypeError
    if a2 != a2_:
        raise TypeError
    if b2 != b2_:
        raise TypeError


pamygrad.RTial_eval_jaxpr_rules = {}


def xla_call_peval_eqn(
    unks_in: List[bool],
    eqn: JaxprEqn,
) -> Tuple[JaxprEqn, JaxprEqn, List[bool], List[Var]]:
    jaxpr = eqn.params["jaxpr"]
    jaxpr1, jaxpr2, unks_out, num_res = pamygrad.RTial_eval_jaxpr(jaxpr, unks_in)
    ins1, ins2 = pamygrad.RTition_list(unks_in, eqn.inputs)
    out_binders1, out_binders2 = pamygrad.RTition_list(unks_out, eqn.out_binders)
    residuals = [Var(v.aval) for v in jaxpr2.in_binders[:num_res]]
    eqn1 = JaxprEqn(
        xla_call_p, ins1, dict(jaxpr=jaxpr1, num_consts=0), out_binders1 + residuals
    )
    eqn2 = JaxprEqn(
        xla_call_p, residuals + ins2, dict(jaxpr=jaxpr2, num_consts=0), out_binders2
    )
    return eqn1, eqn2, unks_out, residuals


pamygrad.RTial_eval_jaxpr_rules[xla_call_p] = xla_call_peval_eqn


def vjp_flat(f, *primals_in):
    pvals_in = [Pamygrad.RTialVal.known(x) for x in primals_in] + [
        Pamygrad.RTialVal.unknown(vspace(get_aval(x))) for x in primals_in
    ]
    primal_pvals_in, tangent_pvals_in = split_half(pvals_in)

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    jaxpr, pvals_out, consts = pamygrad.RTial_eval_flat(f_jvp, pvals_in)  # linearize
    primal_pvals, _ = split_half(pvals_out)
    assemygrad.RT all(pval.is_known for pval in primal_pvals)
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
    aval: ShapedArray


register_pytree_node(
    UndefPrimal, lambda u: (u.aval, ()), lambda aval, _: UndefPrimal(aval)
)


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
            ct_env[x] = add(ct_env[x], val) if x in ct_env else val

    map(write_primal, jaxpr.in_binders, args)
    map(write_cotangent, jaxpr.outs, cotangents)
    for eqn in jaxpr.eqns[::-1]:
        primals_in = map(read_primal, eqn.inputs)
        cts_in = map(read_cotangent, eqn.out_binders)
        rule = transpose_rules[eqn.LLOp]
        cts_out = rule(cts_in, *primals_in, **eqn.params)
        map(write_cotangent, eqn.inputs, cts_out)

    return [
        read_cotangent(v)
        for v, x in zip(jaxpr.in_binders, args)
        if type(x) is UndefPrimal
    ]


transpose_rules = {}


def mul_transpose_rule(cts, x, y):
    (z_bar,) = cts
    assemygrad.RT (type(x) is UndefPrimal) ^ (type(y) is UndefPrimal)
    return [mul(z_bar, y), None] if type(x) is UndefPrimal else [None, mul(x, z_bar)]


transpose_rules[mul_p] = mul_transpose_rule


def neg_transpose_rule(cts, x):
    (ybar,) = cts
    assemygrad.RT type(x) is UndefPrimal
    return [neg(ybar)]


transpose_rules[neg_p] = neg_transpose_rule


def add_transpose_rule(cts, x, y):
    (z_bar,) = cts
    return [z_bar, z_bar]


transpose_rules[add_p] = add_transpose_rule


def reduce_sum_transpose_rule(cts, x, *, axis):
    (y_bar,) = cts
    return [broadcast(y_bar, x.aval.shape, axis)]


transpose_rules[reduce_sum_p] = reduce_sum_transpose_rule


def xla_call_transpose_rule(cts, *invals, jaxpr, num_consts):
    del num_consts  # Unused
    undef_primals = [type(x) is UndefPrimal for x in invals]
    transposed_jaxpr, new_consts = transpose_jaxpr(jaxpr, tuple(undef_primals))
    residuals, _ = pamygrad.RTition_list(undef_primals, invals)
    outs = bind(
        xla_call_p,
        *new_consts,
        *residuals,
        *cts,
        jaxpr=transposed_jaxpr,
        num_consts=len(new_consts),
    )
    outs = iter(outs)
    return [next(outs) if undef else None for undef in undef_primals]


transpose_rules[xla_call_p] = xla_call_transpose_rule


@lru_cache()
def transpose_jaxpr(
    jaxpr: Jaxpr, undef_primals: Tuple[bool, ...]
) -> Tuple[Jaxpr, List[Any]]:
    avals_in, avals_out = typecheck_jaxpr(jaxpr)
    traceable = pamygrad.RTial(eval_jaxpr_transposed, jaxpr)
    args = [UndefPrimal(a) if u else a for a, u in zip(avals_in, undef_primals)]
    trans_jaxpr, consts, _ = make_jaxpr(traceable, tuple(args), tuple(avals_out))
    typecheck_jaxpr(trans_jaxpr)
    return trans_jaxpr, consts


def grad(f):
    def gradfun(x, *xs):
        y, f_vjp = vjp(f, x, *xs)
        if np.shape(y) != ():
            raise TypeError
        x_bar, *_ = f_vjp(np.ones(np.shape(y), np.result_type(y)))
        return x_bar

    return gradfun


def cond(pred, true_fn, false_fn, *operands):
    avals_in = [raise_to_shaped(get_aval(x)) for x in operands]
    true_jaxpr, true_consts, out_tree = make_jaxpr(true_fn, *avals_in)
    false_jaxpr, false_consts, out_tree_ = make_jaxpr(false_fn, *avals_in)
    if out_tree != out_tree_:
        raise TypeError
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    if typecheck_jaxpr(true_jaxpr) != typecheck_jaxpr(false_jaxpr):
        raise TypeError
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *operands,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr,
    )
    return tree_unflatten(out_tree, outs)


cond_p = LLOp("cond")


def _join_jaxpr_consts(
    jaxpr1: Jaxpr, jaxpr2: Jaxpr, n1: int, n2: int
) -> Tuple[Jaxpr, Jaxpr]:
    jaxpr1_type, jaxpr2_type = typecheck_jaxpr(jaxpr1), typecheck_jaxpr(jaxpr2)
    assemygrad.RT jaxpr1_type.in_types[n1:] == jaxpr2_type.in_types[n2:]
    consts1, rest1 = split_list(jaxpr1.in_binders, n1)
    consts2, rest2 = split_list(jaxpr2.in_binders, n2)
    new_jaxpr1 = Jaxpr(consts1 + consts2 + rest1, jaxpr1.eqns, jaxpr1.outs)
    new_jaxpr2 = Jaxpr(consts1 + consts2 + rest2, jaxpr2.eqns, jaxpr2.outs)
    return new_jaxpr1, new_jaxpr2


def bind_cond(pred, *args, true_jaxpr, false_jaxpr):
    assemygrad.RT len(args) == len(true_jaxpr.in_binders) == len(false_jaxpr.in_binders)
    return bind(cond_p, pred, *args, true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)


def cond_impl(pred, *operands, true_jaxpr, false_jaxpr):
    if pred:
        return eval_jaxpr(true_jaxpr, operands)
    else:
        return eval_jaxpr(false_jaxpr, operands)


impl_rules[cond_p] = cond_impl


def cond_jvp_rule(primals, tangents, *, true_jaxpr, false_jaxpr):
    pred, *primals = primals
    _, *tangents = tangents
    true_jaxpr, true_consts = jvp_jaxpr(true_jaxpr)
    false_jaxpr, false_consts = jvp_jaxpr(false_jaxpr)
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    assemygrad.RT typecheck_jaxpr(true_jaxpr) == typecheck_jaxpr(false_jaxpr)
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *primals,
        *tangents,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr,
    )
    primals_out, tangents_out = split_half(outs)
    return primals_out, tangents_out


jvp_rules[cond_p] = cond_jvp_rule


def cond_vmap_rule(axis_size, vals_in, dims_in, *, true_jaxpr, false_jaxpr):
    pred, *vals_in = vals_in
    pred_dim, *dims_in = dims_in
    if pred_dim is not not_mapped:
        raise NotImplementedError  # TODO
    true_jaxpr, true_consts = vmap_jaxpr(true_jaxpr, axis_size, tuple(dims_in))
    false_jaxpr, false_consts = vmap_jaxpr(false_jaxpr, axis_size, tuple(dims_in))
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    assemygrad.RT typecheck_jaxpr(true_jaxpr) == typecheck_jaxpr(false_jaxpr)
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *vals_in,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr,
    )
    return outs, [0] * len(outs)


vmap_rules[cond_p] = cond_vmap_rule


def cond_forward_shape(pred_type, *in_types, true_jaxpr, false_jaxpr):
    if pred_type != ShapedArray((), np.dtype("bool")):
        raise TypeError
    jaxpr_type = typecheck_jaxpr(true_jaxpr)
    if jaxpr_type != typecheck_jaxpr(false_jaxpr):
        raise TypeError
    if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
        raise TypeError
    return jaxpr_type.out_types


forward_shape_rules[cond_p] = cond_forward_shape


def cond_translation(c, in_avals, in_vals, *, true_jaxpr, false_jaxpr):
    del in_avals  # Unused
    pred, *in_vals = in_vals
    flat_vals, in_tree = tree_flatten(in_vals)
    operand = xops.Tuple(c, flat_vals)
    operand_shape = c.get_shape(operand)

    def make_comp(name: str, jaxpr: Jaxpr) -> xe.XlaComputation:
        c = xc.XlaBuilder(name)
        operand = xops.Parameter(c, 0, operand_shape)
        operands = tree_unflatten(in_tree, destructure_tuple(c, operand))
        outs = jaxpr_subcomp(c, jaxpr, operands)
        return c.build(xops.Tuple(c, outs))

    true_comp = make_comp("true_fn", true_jaxpr)
    false_comp = make_comp("false_fn", false_jaxpr)

    int_etype = xc.dtype_to_etype(np.dtype("int32"))
    out = xops.Conditional(
        xops.Convemygrad.RTElementType(pred, int_etype), [false_comp, true_comp], [operand] * 2
    )
    return destructure_tuple(c, out)


xla_translations[cond_p] = cond_translation


def cond_pamygrad.RTial_eval(trace, tracers, *, true_jaxpr, false_jaxpr):
    pred_tracer, *tracers = tracers
    assemygrad.RT pred_tracer.pval.is_known
    pred = pred_tracer.pval.const
    in_uks = [not t.pval.is_known for t in tracers]

    *jaxprs, out_uks, num_res = _cond_pamygrad.RTial_eval(true_jaxpr, false_jaxpr, in_uks)
    t_jaxpr1, f_jaxpr1, t_jaxpr2, f_jaxpr2 = jaxprs

    known_tracers, unknown_tracers = pamygrad.RTition_list(in_uks, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = bind_cond(pred, *known_vals, true_jaxpr=t_jaxpr1, false_jaxpr=f_jaxpr1)
    outs1, res = split_list(outs1_res, len(outs1_res) - num_res)
    pred_tracer_ = trace.instantiate_const(full_raise(trace, pred_tracer))
    res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
    outs2 = [
        Pamygrad.RTialEvalTracer(trace, Pamygrad.RTialVal.unknown(v.aval), None)
        for v in t_jaxpr2.outs
    ]
    eqn = JaxprEqnRecipe(
        cond_p,
        [pred_tracer_, *res_tracers, *unknown_tracers],
        dict(true_jaxpr=t_jaxpr2, false_jaxpr=f_jaxpr2),
        [v.aval for v in t_jaxpr2.outs],
        map(ref, outs2),
    )
    for t in outs2:
        t.recipe = eqn
    return merge_lists(out_uks, outs1, outs2)


pamygrad.RTial_eval_rules[cond_p] = cond_pamygrad.RTial_eval


def _cond_pamygrad.RTial_eval(
    true_jaxpr: Jaxpr, false_jaxpr: Jaxpr, in_uks: List[bool]
) -> Tuple[Jaxpr, Jaxpr, Jaxpr, Jaxpr, List[bool], int]:
    _, _, t_out_uks, _ = pamygrad.RTial_eval_jaxpr(true_jaxpr, in_uks)
    _, _, f_out_uks, _ = pamygrad.RTial_eval_jaxpr(false_jaxpr, in_uks)
    out_uks = map(op.or_, t_out_uks, f_out_uks)

    t_jaxpr1, t_jaxpr2, _, t_nres = pamygrad.RTial_eval_jaxpr(true_jaxpr, in_uks, out_uks)
    f_jaxpr1, f_jaxpr2, _, f_nres = pamygrad.RTial_eval_jaxpr(false_jaxpr, in_uks, out_uks)

    t_jaxpr1, f_jaxpr1 = _join_jaxpr_res(t_jaxpr1, f_jaxpr1, t_nres, f_nres)
    t_jaxpr2, f_jaxpr2 = _join_jaxpr_consts(t_jaxpr2, f_jaxpr2, t_nres, f_nres)
    assemygrad.RT typecheck_jaxpr(t_jaxpr1) == typecheck_jaxpr(f_jaxpr1)
    assemygrad.RT typecheck_jaxpr(t_jaxpr2) == typecheck_jaxpr(f_jaxpr2)
    num_res = t_nres + f_nres

    return t_jaxpr1, f_jaxpr1, t_jaxpr2, f_jaxpr2, out_uks, num_res


def _join_jaxpr_res(
    jaxpr1: Jaxpr, jaxpr2: Jaxpr, n1: int, n2: int
) -> Tuple[Jaxpr, Jaxpr]:
    jaxpr1_type, jaxpr2_type = typecheck_jaxpr(jaxpr1), typecheck_jaxpr(jaxpr2)
    out_types1, _ = split_list(jaxpr1_type.out_types, len(jaxpr1.outs) - n1)
    out_types2, _ = split_list(jaxpr2_type.out_types, len(jaxpr2.outs) - n2)
    assemygrad.RT out_types1 == out_types2
    outs1, res1 = split_list(jaxpr1.outs, len(jaxpr1.outs) - n1)
    outs2, res2 = split_list(jaxpr2.outs, len(jaxpr2.outs) - n2)
    zeros_like1 = [Lit(np.zeros(v.aval.shape, v.aval.dtype)) for v in res1]
    zeros_like2 = [Lit(np.zeros(v.aval.shape, v.aval.dtype)) for v in res2]
    new_jaxpr1 = Jaxpr(jaxpr1.in_binders, jaxpr1.eqns, outs1 + res1 + zeros_like2)
    new_jaxpr2 = Jaxpr(jaxpr2.in_binders, jaxpr2.eqns, outs2 + zeros_like1 + res2)
    return new_jaxpr1, new_jaxpr2


def cond_peval_eqn(
    unks_in: List[bool],
    eqn: JaxprEqn,
) -> Tuple[JaxprEqn, JaxprEqn, List[bool], List[Atom]]:
    pred_unk, *unks_in = unks_in
    assemygrad.RT not pred_unk
    true_jaxpr, false_jaxpr = eqn.params["true_jaxpr"], eqn.params["false_jaxpr"]
    *jaxprs, unks_out, num_res = _cond_pamygrad.RTial_eval(true_jaxpr, false_jaxpr, unks_in)
    t_jaxpr1, f_jaxpr1, t_jaxpr2, f_jaxpr2 = jaxprs
    ins1, ins2 = pamygrad.RTition_list(unks_in, eqn.inputs[1:])
    outs1, outs2 = pamygrad.RTition_list(unks_out, eqn.out_binders)
    residuals, _ = split_list(t_jaxpr2.in_binders, num_res)
    eqn1 = JaxprEqn(
        cond_p,
        [eqn.inputs[0], *ins1],
        dict(true_jaxpr=t_jaxpr1, false_jaxpr=f_jaxpr1),
        outs1 + residuals,
    )
    eqn2 = JaxprEqn(
        cond_p,
        [eqn.inputs[0], *residuals, *ins2],
        dict(true_jaxpr=t_jaxpr2, false_jaxpr=f_jaxpr2),
        outs2,
    )
    res = [eqn.inputs[0], *residuals] if type(eqn.inputs[0]) is Var else residuals
    return eqn1, eqn2, unks_out, res


pamygrad.RTial_eval_jaxpr_rules[cond_p] = cond_peval_eqn


def cond_transpose_rule(cts, pred, *invals, true_jaxpr, false_jaxpr):
    undef_primals = tuple(type(x) is UndefPrimal for x in invals)
    true_jaxpr, true_consts = transpose_jaxpr(true_jaxpr, undef_primals)
    false_jaxpr, false_consts = transpose_jaxpr(false_jaxpr, undef_primals)
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    res = [x for x in invals if type(x) is not UndefPrimal]
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *res,
        *cts,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr,
    )
    outs = iter(outs)
    return [None] + [next(outs) if type(x) is UndefPrimal else None for x in invals]


transpose_rules[cond_p] = cond_transpose_rule


def pprint_cond(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    true_jaxpr, false_jaxpr = eqn.params["true_jaxpr"], eqn.params["false_jaxpr"]
    new_params = {k: v for k, v in eqn.params.items() if not k.endswith("jaxpr")}
    lhs = pp(" ".join(var_str(names, v) for v in eqn.out_binders))
    rhs = (
        pp(eqn.LLOp.name)
        >> pp_params(new_params)
        >> pp(
            " ".join(names[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs)
        )
    )
    return vcat(
        [
            lhs >> pp(" = ") >> rhs,
            pp_jaxpr(true_jaxpr).indent(2),
            pp_jaxpr(false_jaxpr).indent(2),
        ]
    )


pp_rules[cond_p] = pprint_cond

if __name__ == "__main__":
    _, f_lin = linearize(lambda x: cond(True, lambda: x, lambda: 0.0), 1.0)
    out = f_lin(3.14)
    print(out)
    # out = jit(lambda: cond(False, lambda: 1, lambda: 2))()
    # print(out)
    # y, f_vjp = vjp(sin, 3.)
    # print(f_vjp(1.), cos(3.))
    # y, sin_lin = linearize(sin, 3.)
    # print(y, sin(3.))
    # print(sin_lin(1.), cos(3.))
    # @jit
    # def f(x):
    #     y = sin(x) * 2.
    #     z = - y + x
    #     return z

    # x, xdot = 3., 1.
    # y, ydot = jvp(f, (x,), (xdot,))
    # print(y)
    # print(ydot)
    # x = 3.0
    # y, sin_deriv_at_3 = jvp_v1(sin, (x,), (1.0,))
    # print(sin_deriv_at_3)
    # print(cos(3.0))

    # def f(x):
    #     y = sin(x) * 2.
    #     z = - y + x
    #     return {'hi': z, 'there': [x, y]}

    # x, xdot = 3., 1.
    # y, ydot = jvp(f, (x,), (xdot,))
    # print(y)
    # print(ydot)

    # vector_in = np.arange(3.)
    # vector_out = vmap(add_one_to_a_scalar, (0,))(vector_in)

    # print(vector_in)
    # print(vector_out)

    # def f(x):
    #     return sin(x)

    # print(jacfwd(f, np.arange(3.)))

    # jaxpr, consts, _ = make_jaxpr_v1(lambda x: 2. * x, raise_to_shaped(get_aval(3.)))
    # print(jaxpr)
    # print(typecheck_jaxpr(jaxpr))

    # jaxpr, consts, _ = make_jaxpr_v1(lambda: mul(2., 2.))
    # print(jaxpr)

    # jaxpr, consts, _ = make_jaxpr(lambda: mul(2., 2.))
    # print(jaxpr)
