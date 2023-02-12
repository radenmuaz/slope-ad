from typing import NamedTuple
from contextlib import contextmanager
from typing import Type, List, Tuple, Sequence, Optional, Any
import numpy as np
import operator as op
import itertools as it
from typing import Callable, Type, Hashable, Dict, Iterable, Iterator
from typing import Union
from typing import Set

from functools import partial
from functools import lru_cache

from typing import DefaultDict
from collections import defaultdict
import string

class Primitive(NamedTuple):
    name: str

add_p = Primitive('add')
mul_p = Primitive('mul')
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")
less_p = Primitive("less")
transpose_p = Primitive("transpose")
broadcast_p = Primitive("broadcast")

def add(x, y): return bind1(add_p, x, y)
def mul(x, y): return bind1(mul_p, x, y)
def neg(x): return bind1(neg_p, x)
def sin(x): return bind1(sin_p, x)
def cos(x): return bind1(cos_p, x)
def greater(x, y): return bind1(greater_p, x, y)
def less(x, y): return bind1(less_p, x, y)
def transpose(x, perm): return bind1(transpose_p, x, perm=perm)
def broadcast(x, shape, axes): return bind1(broadcast_p, x, shape=shape, axes=axes)
def reduce_sum(x, axis=None):
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    if type(axis) is int:
        axis = (axis,)
    return bind1(reduce_sum_p, x, axis=axis)

def bind1(prim, *args, **params):
    out, = bind(prim, *args, **params)
    return out

def bind(prim, *args, **params):
  top_trace = find_top_trace(args)
  tracers = [full_raise(top_trace, arg) for arg in args]
  outs = top_trace.process_primitive(prim, tracers, params)
  return [full_lower(out) for out in outs]




class MainTrace(NamedTuple):
    level: int
    trace_type: Type['Trace']
    global_data: Optional[Any]

trace_stack: List[MainTrace] = []
dynamic_trace: Optional[MainTrace] = None  # to be employed in Part 3

@contextmanager
def new_main(trace_type: Type['Trace'], global_data=None):
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

    def pure(self, val): assert False  # must override
    def lift(self, val): assert False  # must override

    def process_primitive(self, primitive, tracers, params):
        assert False  # must override


class Tracer:
    _trace: Trace

    __array_priority__ = 1000

    @property
    def aval(self):
        assert False  # must override

    def full_lower(self):
        return self  # default implementation

    def __neg__(self): return self.aval._neg(self)
    def __add__(self, other): return self.aval._add(self, other)
    def __radd__(self, other): return self.aval._radd(self, other)
    def __mul__(self, other): return self.aval._mul(self, other)
    def __rmul__(self, other): return self.aval._rmul(self, other)
    def __gt__(self, other): return self.aval._gt(self, other)
    def __lt__(self, other): return self.aval._lt(self, other)
    def __bool__(self): return self.aval._bool(self)
    def __nonzero__(self): return self.aval._nonzero(self)

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

def swap(f): return lambda x, y: f(y, x)

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
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  @staticmethod
  def _nonzero(tracer):
    raise Exception("ShapedArray can't be unambiguously converted to bool")

  def str_short(self):
    return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

  def __hash__(self):
    return hash((self.shape, self.dtype))

  def __eq__(self, other):
    return (type(self) is type(other) and
            self.shape == other.shape and self.dtype == other.dtype)

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

jax_types = {bool, int, float,
             np.bool_, np.int32, np.int64, np.float32, np.float64, np.ndarray}


def find_top_trace(xs) -> Trace:
    top_main = max((x._trace.main for x in xs if isinstance(x, Tracer)),
                 default=trace_stack[0], key=op.attrgetter('level'))
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
        assert type(val) in jax_types
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

  def process_primitive(self, primitive, tracers, params):
    return impl_rules[primitive](*tracers, **params)

trace_stack.append(MainTrace(0, EvalTrace, None))  # special bottom of the stack

# NB: in JAX, instead of a dict we attach impl rules to the Primitive instance
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
    for axis in sorted(axes):
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
        assert len(arg) == n
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

    def process_primitive(self, primitive, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        jvp_rule = jvp_rules[primitive]
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
  if in_tree != in_tree2: raise TypeError
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

class Empty: pass
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

def register_pytree_node(ty: Type, to_iter: Callable, from_iter: Callable
                         ) -> None:
  node_types[ty] = NodeType(str(ty), to_iter, from_iter)

node_types: Dict[Type, NodeType] = {}
register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
register_pytree_node(list,  lambda l: (None, l), lambda _, xs:  list(xs))
register_pytree_node(dict,
                     lambda d: map(tuple, unzip2(sorted(d.items()))),
                     lambda keys, vals: dict(zip(keys, vals)))

class PyTreeDef(NamedTuple):
  node_type: NodeType
  node_metadata: Hashable
  child_treedefs: Tuple['PyTreeDef', ...]

class Leaf: pass
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
        target_shape.insert(dst, axis_size)
        return broadcast(x, target_shape, [dst])
    elif src == dst:
        return x
    else:
        return moveaxis(x, src, dst)

def moveaxis(x, src: int, dst: int):
    perm = [i for i in range(np.ndim(x)) if i != src]
    perm.insert(dst, src)
    return transpose(x, perm)

class NotMapped: pass
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

    def process_primitive(self, primitive, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
        vmap_rule = vmap_rules[primitive]
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

vmap_rules[add_p] = partial(binop_batching_rule, add)
vmap_rules[mul_p] = partial(binop_batching_rule, mul)

def vectorized_unop_batching_rule(op, axis_size, vals_in, dims_in):
    (x,), (x_bdim,) = vals_in, dims_in
    return [op(x)], [x_bdim]

vmap_rules[sin_p] = partial(vectorized_unop_batching_rule, sin)
vmap_rules[cos_p] = partial(vectorized_unop_batching_rule, cos)
vmap_rules[neg_p] = partial(vectorized_unop_batching_rule, neg)

def reduce_sum_batching_rule(axis_size, vals_in, dims_in, *, axis):
    (x,), (x_bdim,) = vals_in, dims_in
    new_axis = tuple(ax + (x_bdim <= ax) for ax in axis)
    out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
    return [reduce_sum(x, new_axis)], [out_bdim]

vmap_rules[reduce_sum_p] = reduce_sum_batching_rule

def vmap_flat(f, in_axes, *args):
    axis_size, = {x.shape[ax] for x, ax in zip(args, in_axes)
                if ax is not not_mapped}
    with new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [BatchTracer(trace, x, ax) if ax is not None else x
                    for x, ax in zip(args, in_axes)]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_transposed = [move_batch_axis(axis_size, bdim, 0, val_out)
                     for val_out, bdim in zip(vals_out, bdims_out)]
    return outs_transposed

def vmap(f, in_axes):
    def batched_f(*args):
        args_flat, in_tree = tree_flatten(args)
        in_axes_flat, in_tree2 = tree_flatten(in_axes)
        if in_tree != in_tree2: raise TypeError
        f_flat, out_tree = flatten_fun(f, in_tree)
        outs_flat = vmap_flat(f_flat, in_axes_flat, *args_flat)
        return tree_unflatten(out_tree(), outs_flat)
    return batched_f

def add_one_to_a_scalar(scalar):
    assert np.ndim(scalar) == 0
    return 1 + scalar

def jacfwd(f, x):
    pushfwd = lambda v: jvp(f, (x,), (v,))[1]
    vecs_in = np.eye(np.size(x)).reshape(np.shape(x) * 2)
    return vmap(pushfwd, (0,))(vecs_in)


def raise_to_shaped(aval):
    return ShapedArray(aval.shape, aval.dtype)

class Var:
    aval: ShapedArray
    def __init__(self, aval): self.aval = aval

class Lit:
    val: Any
    aval: ShapedArray

    def __init__(self, val):
        self.aval = aval = raise_to_shaped(get_aval(val))
        self.val = np.array(val, aval.dtype)

Atom = Union[Var, Lit]

class JaxprEqn(NamedTuple):
    primitive: Primitive
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Var]

class Jaxpr(NamedTuple):
    in_binders: List[Var]
    eqns: List[JaxprEqn]
    outs: List[Atom]

    def __hash__(self): return id(self)
    __eq__ = op.is_

class JaxprType(NamedTuple):
    in_types:  List[ShapedArray]
    out_types: List[ShapedArray]

    def __repr__(self):
        in_types = ', '.join(aval.str_short() for aval in self.in_types)
        out_types = ', '.join(aval.str_short() for aval in self.out_types)
        return f'({in_types}) -> ({out_types})'

def typecheck_jaxpr(jaxpr: Jaxpr) -> JaxprType:
    env: Set[Var] = set()

    for v in jaxpr.in_binders:
        if v in env: raise TypeError
        env.add(v)

    for eqn in jaxpr.eqns:
        in_types = [typecheck_atom(env, x) for x in eqn.inputs]
        out_types = abstract_eval_rules[eqn.primitive](*in_types, **eqn.params)
        for out_binder, out_type in zip(eqn.out_binders, out_types):
            if not out_type == out_binder.aval: raise TypeError
        for out_binder in eqn.out_binders:
            if out_binder in env: raise TypeError
            env.add(out_binder)

    in_types = [v.aval for v in jaxpr.in_binders]
    out_types = [typecheck_atom(env, x) for x in jaxpr.outs]
    return JaxprType(in_types, out_types)

def typecheck_atom(env: Set[Var], x: Atom) -> ShapedArray:
    if isinstance(x, Var):
        if x not in env: raise TypeError("unbound variable")
        return x.aval
    elif isinstance(x, Lit):
        return raise_to_shaped(get_aval(x.val))
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
        outs = bind(eqn.primitive, *in_vals, **eqn.params)
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

class JaxprTracer(Tracer):
    __slots__ = ['aval']
    aval: ShapedArray

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval

class JaxprTrace(Trace):
    def new_arg(self, aval: ShapedArray) -> JaxprTracer:
        aval = raise_to_shaped(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)
        return tracer

    def get_or_make_const_tracer(self, val: Any) -> JaxprTracer:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, raise_to_shaped(get_aval(val)))
            self.builder.add_const(tracer, val)
        return tracer
    pure = lift = get_or_make_const_tracer

    def process_primitive(self, primitive, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = abstract_eval_rules[primitive](*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_eqn(JaxprEqn(primitive, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data

abstract_eval_rules = {}

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

    def new_tracer(self, trace: JaxprTrace, aval: ShapedArray) -> JaxprTracer:
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

    def build(self, in_tracers: List[JaxprTracer], out_tracers: List[JaxprTracer]
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
    new_const_binders, lit_binders = partition_list(scalars, const_binders)
    new_consts, lit_vals = partition_list(scalars, consts)
    literals = dict(zip(lit_binders, map(Lit, lit_vals)))
    new_eqns = [JaxprEqn(eqn.primitive, [literals.get(x, x) for x in eqn.inputs],
                        eqn.params, eqn.out_binders) for eqn in jaxpr.eqns]
    new_outs = [literals.get(x, x) for x in jaxpr.outs]
    new_jaxpr = Jaxpr(new_const_binders + other_binders, new_eqns, new_outs)
    typecheck_jaxpr(new_jaxpr)
    return new_jaxpr, new_consts

def binop_abstract_eval(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
    if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
        raise TypeError
    if raise_to_shaped(x) != raise_to_shaped(y): raise TypeError
    return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[add_p] = binop_abstract_eval
abstract_eval_rules[mul_p] = binop_abstract_eval

def compare_abstract_eval(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
    if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
        raise TypeError
    if x.shape != y.shape: raise TypeError
    return [ShapedArray(x.shape, np.dtype('bool'))]
    abstract_eval_rules[greater_p] = compare_abstract_eval
    abstract_eval_rules[less_p] = compare_abstract_eval

def vectorized_unop_abstract_eval(x: ShapedArray) -> List[ShapedArray]:
    return [ShapedArray(x.shape, x.dtype)]

abstract_eval_rules[sin_p] = vectorized_unop_abstract_eval
abstract_eval_rules[cos_p] = vectorized_unop_abstract_eval
abstract_eval_rules[neg_p] = vectorized_unop_abstract_eval

def reduce_sum_abstract_eval(x: ShapedArray, *, axis: Tuple[int, ...]
                             ) -> List[ShapedArray]:
    axis_ = set(axis)
    new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
    return [ShapedArray(tuple(new_shape), x.dtype)]
abstract_eval_rules[reduce_sum_p] = reduce_sum_abstract_eval

def broadcast_abstract_eval(x: ShapedArray, *, shape: Sequence[int],
                            axes: Sequence[int]) -> List[ShapedArray]:
    return [ShapedArray(tuple(shape), x.dtype)]
abstract_eval_rules[broadcast_p] = broadcast_abstract_eval


@lru_cache()  # ShapedArrays are hashable
def make_jaxpr_v1(f, *avals_in):
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)

    builder = JaxprBuilder()
    with new_main(JaxprTrace, builder) as main:
        trace = JaxprTrace(main)
        tracers_in = [trace.new_arg(aval) for aval in avals_in]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        jaxpr, consts = builder.build(tracers_in, tracers_out)
    return jaxpr, consts, out_tree()


class PPrint:
    lines: List[Tuple[int, str]]

    def __init__(self, lines):
        self.lines = lines

    def indent(self, indent: int) -> 'PPrint':
        return PPrint([(indent + orig_indent, s) for orig_indent, s in self.lines])

    def __add__(self, rhs: 'PPrint') -> 'PPrint':
        return PPrint(self.lines + rhs.lines)

    def __rshift__(self, rhs: 'PPrint') -> 'PPrint':
        if not rhs.lines: return self
        if not self.lines: return rhs
        indent, s = self.lines[-1]
        indented_block = rhs.indent(indent + len(s))
        common_line = s + ' ' * rhs.lines[0][0] + rhs.lines[0][1]
        return PPrint(self.lines[:-1]
                    + [(indent, common_line)]
                    + indented_block.lines[1:])

    def __str__(self) -> str:
        return '\n'.join(' ' * indent + s for indent, s in self.lines)

def pp(s: Any) -> PPrint:
    return PPrint([(0, line) for line in str(s).splitlines()])

def vcat(ps: List[PPrint]) -> PPrint:
    return sum(ps, pp(''))

def pp_jaxpr(jaxpr: Jaxpr) -> PPrint:
    namegen = (''.join(s) for r in it.count(1)
            for s in it.permutations(string.ascii_lowercase, r))
    names = defaultdict(lambda: next(namegen))
    in_binders = ', '.join(var_str(names, x) for x in jaxpr.in_binders)
    eqns = vcat([pp_eqn(names, e) for e in jaxpr.eqns])
    outs = ', '.join(names[v] if isinstance(v, Var) else str(v.val)
                for v in jaxpr.outs)
    return (pp(f'{{ lambda {in_binders} .') +
          ((pp('let ') >> eqns) + pp(f'in ( {outs} ) }}')).indent(2))

def var_str(names: DefaultDict[Var, str], v: Var) -> str:
    return f'{names[v]}:{v.aval.str_short()}'

def pp_eqn(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    rule = pp_rules.get(eqn.primitive)
    if rule:
        return rule(names, eqn)
    else:
        lhs = pp(' '.join(var_str(names, v) for v in eqn.out_binders))
        rhs = (pp(eqn.primitive.name) >> pp_params(eqn.params) >>
           pp(' '.join(names[x] if isinstance(x, Var) else str(x.val)
                       for x in eqn.inputs)))
        return lhs >> pp(' = ') >> rhs

def pp_params(params: Dict[str, Any]) -> PPrint:
    items = sorted(params.items())
    if items:
        return pp(' [ ') >> vcat([pp(f'{k}={v}') for k, v in items]) >> pp(' ] ')
    else:
        return pp(' ')

Jaxpr.__repr__ = lambda self: str(pp_jaxpr(self))
pp_rules: Dict[Primitive, Callable[..., PPrint]] = {}

@contextmanager
def new_dynamic(main: MainTrace):
    global dynamic_trace
    prev_dynamic_trace, dynamic_trace = dynamic_trace, main
    try:
        yield
    finally:
        dynamic_trace = prev_dynamic_trace

@lru_cache()
def make_jaxpr(f: Callable, *avals_in: ShapedArray,
               ) -> Tuple[Jaxpr, List[Any], PyTreeDef]:
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)

    builder = JaxprBuilder()
    with new_main(JaxprTrace, builder) as main:
        with new_dynamic(main):
        trace = JaxprTrace(main)
        tracers_in = [trace.new_arg(aval) for aval in avals_in]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        jaxpr, consts = builder.build(tracers_in, tracers_out)
    return jaxpr, consts, out_tree()

if __name__ == "__main__":
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

    jaxpr, consts, _ = make_jaxpr(lambda: mul(2., 2.))
    print(jaxpr)

