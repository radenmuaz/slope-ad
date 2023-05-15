from contextlib import contextmanager
from slope import utils
import numpy as np
import itertools
import weakref
from typing import (
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
from functools import lru_cache, reduce

import slope
from slope.array_shape import ArrayShape, ValuedArrayShape
from slope import ops
# from slope.array import Array

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

    @classmethod
    def pp(cls, s: Any):
        return cls([(0, line) for line in str(s).splitlines()])

    @classmethod
    def vcat(cls, ps: List["PPrint"]):
        return sum(ps, cls.pp(""))


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
        node_type = slope.RT.node_types.get(type(x))

        if node_type:
            node_metadata, children = node_type.to_iterable(x)
            children_flat, child_trees = utils.unzip2(
                utils.list_map(_tree_flatten, children)
            )
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

class EvalTrace(Trace):
    pure = lift = lambda self, x: x

    def run_op(self, op, tracers, params):
        return op.eval(*tracers, **params)


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

    @staticmethod
    def get_aval(x):
        if isinstance(x, Tracer):
            return x.aval
        elif type(x) in Tracer.TYPES:
            return ValuedArrayShape(np.asarray(x))
        else:
            raise TypeError(x)

    @property
    def aval(self):
        assert False  # must override

    def full_lower(self):
        return self  # default implementation

    def __neg__(self):
        return self.aval._neg(self)

    def __add__(self, other):
        return slope.RT.bind1(ops.Add, self, other)

    def __radd__(self, other):
        return slope.RT.bind1(ops.Add, other, self)

    def __mul__(self, other):
        return slope.RT.bind1(ops.Mul, self, other)

    def __rmul__(self, other):
        return slope.RT.bind1(ops.Mul, other, self)

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")



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
            return slope.RT.full_lower(self.val)
        else:
            return self


class BatchTrace(Trace):
    pure = lift = lambda self, val: BatchTracer(self, val, None)

    def run_op(self, op, tracers, params):
        vals_in, bdims_in = utils.unzip2((t.val, t.batch_dim) for t in tracers)
        val_outs, bdim_outs = op.vmap(self.axis_size, vals_in, bdims_in, **params)
        return [
            BatchTracer(self, x, bd) for x, bd in utils.list_zip(val_outs, bdim_outs)
        ]

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
        reshape_shape = [1 if ax == dst else target_shape for ax in range(out_ndim)]
        x = ops.reshape(x, reshape_shape)
        x = ops.broadcast(x, target_shape)
        return x
    elif src == dst:
        return x
    else:
        perm = [i for i in range(len(x.shape)) if i != src]
        perm.insert(dst, src)
        return ops.transpose(x, perm)


def vmap_flat(f, in_axes, *args):
    axis_set = {
        x.shape[ax] for x, ax in utils.list_zip(args, in_axes) if ax is not None
    }
    assert len(axis_set) == 1
    (axis_size,) = axis_set
    with slope.RT.new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [
            BatchTracer(trace, x, ax) if ax is not None else x
            for x, ax in utils.list_zip(args, in_axes)
        ]
        outs = f(*tracers_in)
        tracers_out = [slope.RT.full_raise(trace, out) for out in outs]
        vals_out, bdims_out = utils.unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_transposed = [
        move_batch_axis(axis_size, bdim, 0, val_out)
        for val_out, bdim in utils.list_zip(vals_out, bdims_out)
    ]
    return outs_transposed


def vmap(f, in_axes):
    def batched_f(*args):
        args_flat, in_tree = tree_flatten(args)
        in_axes_flat, in_tree2 = tree_flatten(in_axes)
        if in_tree != in_tree2:
            raise TypeError(f"{in_tree}\n!=\n{in_tree2}")
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
        return JVPTracer(self, val, np.zeros(aval.shape, aval.dtype))

    lift = pure

    def run_op(self, op, tracers, params):
        primals_in, tangents_in = utils.unzip2((t.primal, t.tangent) for t in tracers)
        
        primal_outs, tangent_outs = op.jvp(primals_in, tangents_in, **params)
        return [
            JVPTracer(self, x, t) for x, t in utils.list_zip(primal_outs, tangent_outs)
        ]


def jvp_flat(f, primals, tangents):
    with slope.RT.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [
            JVPTracer(trace, x, t) for x, t in utils.list_zip(primals, tangents)
        ]
        outs = f(*tracers_in)
        tracers_out = [slope.RT.full_raise(trace, out) for out in outs]
        primals_out, tangents_out = utils.unzip2(
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


class Instruction(NamedTuple):
    op: ops.Op
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Atom]


class Program(NamedTuple):
    in_binders: Any
    instructions: List[Instruction]
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
        in_binders = ", ".join(self.var_str(names, x) for x in self.in_binders)
        instruction = PPrint.vcat([self.pp_eqn(names, e) for e in self.instruction])
        outs = ", ".join(
            names[v] if isinstance(v, Var) else str(v.val) for v in self.outs
        )
        return str(
            PPrint.pp(f"{{ lambda {in_binders} .")
            + ((PPrint.pp("let ") >> instruction) + PPrint.pp(f"in ( {outs} ) }}")).indent(2)
        )

    def pp_eqn(self, names: DefaultDict[Var, str], eqn: Instruction) -> PPrint:
        # rule = eqn.op.pprint
        # if rule() is not None:
        #     return rule(names, eqn)
        # else:
        lhs = PPrint.pp(" ".join(self.var_str(names, v) for v in eqn.out_binders))
        rhs = (
            PPrint.pp(repr(eqn.op.__class__))
            >> self.pp_params(eqn.params)
            >> PPrint.pp(
                " ".join(
                    names[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs
                )
            )
        )
        return lhs >> PPrint.pp(" = ") >> rhs

    def pp_params(self, params: Dict[str, Any]) -> PPrint:
        items = sorted(params.items())
        if items:
            return (
                PPrint.pp(" [ ")
                >> PPrint.vcat([PPrint.pp(f"{k}={v}") for k, v in items])
                >> PPrint.pp(" ] ")
            )
        else:
            return PPrint.pp(" ")

    def var_str(self, names: DefaultDict[Var, str], v) -> str:
        return f"{names[v]}:{v.aval.str_short()}"


class ProgramType(NamedTuple):
    in_types: List[ArrayShape]
    out_types: List[ArrayShape]

    def __repr__(self):
        in_types = ", ".join(aval.str_short() for aval in self.in_types)
        out_types = ", ".join(aval.str_short() for aval in self.out_types)
        return f"({in_types}) -> ({out_types})"


def typecheck_program(Program: Program) -> ProgramType:
    env: Set[Var] = set()

    for v in Program.in_binders:
        if v in env:
            raise TypeError
        env.add(v)

    for eqn in Program.instructions:
        in_types = [typecheck_atom(env, x) for x in eqn.inputs]
        out_types = eqn.op.shape_eval(*in_types, **eqn.params)
        for out_binder, out_type in utils.list_zip(eqn.out_binders, out_types):
            if not out_type == out_binder.aval:
                raise TypeError
        for out_binder in eqn.out_binders:
            if out_binder in env:
                raise TypeError
            env.add(out_binder)

    in_types = [v.aval for v in Program.in_binders]
    out_types = [typecheck_atom(env, x) for x in Program.outs]
    return ProgramType(in_types, out_types)


def typecheck_atom(env: Set[Var], x: Atom) -> ArrayShape:
    if isinstance(x, Var):
        if x not in env:
            raise TypeError("unbound variable")
        return x.aval
    elif isinstance(x, Lit):
        return Tracer.get_aval(x.val)
    else:
        assert False


def eval_program(Program: Program, args: List[Any]) -> List[Any]:
    env: Dict[Var, Any] = {}

    def read(x: Atom) -> Any:
        return env[x] if type(x) is Var else x.val

    def write(v: Var, val: Any) -> None:
        assert v not in env  # single-assignment
        env[v] = val

    utils.list_map(write, Program.in_binders, args)
    for eqn in Program.instruction:
        in_vals = utils.list_map(read, eqn.inputs)
        outs = slope.RT.bind(eqn.op, *in_vals, **eqn.params)
        utils.list_map(write, eqn.out_binders, outs)
    return utils.list_map(read, Program.outs)


def Program_as_fun(Program: Program):
    return lambda *args: eval_program(Program, args)


class ProgramTracer(Tracer):
    __slots__ = ["aval"]
    aval: ArrayShape

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval


class ProgramTrace(Trace):
    def new_arg(self, aval) -> ProgramTracer:
        aval = ArrayShape.like(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)

        return tracer

    def get_or_make_const_tracer(self, val: Any) -> ProgramTracer:
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
        self.builder.add_eqn(Instruction(op, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data


class ProgramBuilder:
    instruction: List[Instruction]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, Tracer]
    constvals: Dict[Var, Any]
    tracers: List[ProgramTracer]

    def __init__(self):
        self.instruction = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    def new_tracer(self, trace: ProgramTrace, aval: ArrayShape) -> ProgramTracer:
        tracer = ProgramTracer(trace, aval)
        self.tracers.append(tracer)
        return tracer

    def add_eqn(self, eqn: Instruction) -> None:
        self.instruction.append(eqn)

    def add_var(self, tracer: ProgramTracer) -> Var:
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
        return var

    def getvar(self, tracer: ProgramTracer) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer: ProgramTracer, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(self, in_tracers: Any, out_tracers: Any) -> Tuple[Program, List[Any]]:
        constvars, constvals = utils.unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        Program = Program(in_binders, self.instruction, out_vars)
        typecheck_program(Program)
        Program, constvals = self._inline_literals(Program, constvals)
        return Program, constvals

    def _inline_literals(
        self, program: Program, consts: List[Any]
    ) -> Tuple[Program, List[Any]]:
        const_binders, other_binders = utils.split_list(program.in_binders, len(consts))
        scalars = [
            type(x) in Tracer.TYPES and not Tracer.get_aval(x).shape for x in consts
        ]
        new_const_binders, lit_binders = utils.partition_list(scalars, const_binders)
        new_consts, lit_vals = utils.partition_list(scalars, consts)
        literals = dict(zip(lit_binders, utils.list_map(Lit, lit_vals)))
        new_instruction = [
            Instruction(
                eqn.op,
                [literals.get(x, x) for x in eqn.inputs],
                eqn.params,
                eqn.out_binders,
            )
            for eqn in program.instruction
        ]
        new_outs = [literals.get(x, x) for x in program.outs]
        new_program = program(new_const_binders + other_binders, new_instruction, new_outs)
        typecheck_program(new_program)
        return new_program, new_consts


@lru_cache()
def make_program(
    f: Callable,
    *avals_in: ArrayShape,
) -> Tuple[Program, List[Any], PyTreeDef]:
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)

    builder = ProgramBuilder()
    with slope.RT.new_main(ProgramTrace, builder) as main:
        with slope.RT.new_dynamic(main):
            trace = ProgramTrace(main)
            tracers_in = [trace.new_arg(aval) for aval in avals_in]
            outs = f(*tracers_in)
            tracers_out = [slope.RT.full_raise(trace, out) for out in outs]
            program, consts = builder.build(tracers_in, tracers_out)
    return program, consts, out_tree()


def linearize_flat(f, *primals_in):
    pvals_in = [PartialVal.known(x) for x in primals_in] + [
        PartialVal.unknown(ArrayShape.like(Tracer.get_aval(x))) for x in primals_in
    ]

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *utils.split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    program, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)
    primal_pvals, _ = utils.split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    f_lin = lambda *tangents: eval_program(program, [*consts, *tangents])
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
) -> Tuple[Program, List[PartialVal], List[Any]]:
    with slope.RT.new_main(PartialEvalTrace) as main:
        trace = PartialEvalTrace(main)
        tracers_in = [trace.new_arg(pval) for pval in pvals_in]
        outs = f(*tracers_in)
        tracers_out = [slope.RT.full_raise(trace, out) for out in outs]
        pvals_out = [t.pval for t in tracers_out]
        unk_tracers_in = [t for t in tracers_in if t.pval.is_unknown]
        unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
        Program, consts = tracers_to_program(unk_tracers_in, unk_tracers_out)

    return Program, pvals_out, consts


class LambdaBindingProto(NamedTuple):
    pass


class ConstProto(NamedTuple):
    val: Any


class InstructionProto(NamedTuple):
    prim: ops.Op
    tracers_in: List["PartialEvalTracer"]
    params: Dict[str, Any]
    avals_out: List[ArrayShape]
    # tracer_refs_out: List[weakref.ReferenceType["PartialEvalTracer"]]
    tracer_refs_out: List


ProgramProto = Union[LambdaBindingProto, ConstProto, InstructionProto]


class PartialEvalTracer(Tracer):
    pval: PartialVal
    proto: Optional[ProgramProto]

    def __init__(self, trace, pval, proto):
        self._trace = trace
        self.pval = pval
        self.proto = proto

    aval = property(lambda self: self.pval.aval)

    def full_lower(self):
        if self.pval.is_known:
            return slope.RT.full_lower(self.pval.const)
        return self


class PartialEvalTrace(Trace):
    def new_arg(self, pval: PartialVal) -> Any:
        return PartialEvalTracer(self, pval, LambdaBindingProto())

    def lift(self, val: Any) -> PartialEvalTracer:
        return PartialEvalTracer(self, PartialVal.known(val), None)

    pure = lift

    def instantiate_const(self, tracer: PartialEvalTracer) -> PartialEvalTracer:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = PartialVal.unknown(ArrayShape.like(tracer.aval))
            return PartialEvalTracer(self, pval, ConstProto(tracer.pval.const))

    def run_op(self, op, tracers, params):
        if all(t.pval.is_known for t in tracers):
            return slope.RT.bind(op, *map(slope.RT.full_lower, tracers), **params)
        tracers_in = [self.instantiate_const(t) for t in tracers]
        avals_in = [t.aval for t in tracers_in]
        avals_out = op.shape_eval(*avals_in, **params)
        tracers_out = [
            PartialEvalTracer(self, PartialVal.unknown(aval), None)
            for aval in avals_out
        ]
        eqn = InstructionProto(
            op, tracers_in, params, avals_out, map(weakref.ref, tracers_out)
        )
        for t in tracers_out:
            t.proto = eqn
        return tracers_out


def tracers_to_program(
    tracers_in: List[PartialEvalTracer], tracers_out: List[PartialEvalTracer]
):
    tracer_to_var: Dict[int, Var] = {
        id(t): Var(ArrayShape.like(t.aval)) for t in tracers_in
    }
    constvar_to_val: Dict[int, Any] = {}
    constid_to_var: Dict[int, Var] = {}
    processed_instruction: Set[int] = set()
    instruction: List[Instruction] = []
    for t in toposort(tracers_out, tracer_parents):
        if isinstance(t.proto, LambdaBindingProto):
            assert id(t) in set(utils.list_map(id, tracers_in))
        elif isinstance(t.proto, ConstProto):
            val = t.proto.val
            var = constid_to_var.get(id(val))
            if var is None:
                aval = ArrayShape.like(Tracer.get_aval(val))
                var = constid_to_var[id(val)] = Var(aval)
                constvar_to_val[var] = val
            tracer_to_var[id(t)] = var
        elif isinstance(t.proto, InstructionProto):
            if id(t.proto) not in processed_instruction:
                instruction.append(proto_to_eqn(tracer_to_var, t.proto))
                processed_instruction.add(id(t.proto))
        else:
            raise TypeError(t.proto)

    constvars, constvals = utils.unzip2(constvar_to_val.items())
    in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
    out_vars = [tracer_to_var[id(t)] for t in tracers_out]
    program = Program(in_binders, instruction, out_vars)
    typecheck_program(program)
    return program, constvals


def proto_to_eqn(tracer_to_var: Dict[int, Var], proto: InstructionProto) -> Instruction:
    inputs = [tracer_to_var[id(t)] for t in proto.tracers_in]
    out_binders = [Var(aval) for aval in proto.avals_out]
    for t_ref, var in zip(proto.tracer_refs_out, out_binders):
        if t_ref() is not None:
            tracer_to_var[id(t_ref())] = var
    return Instruction(proto.prim, inputs, proto.params, out_binders)


def tracer_parents(t: PartialEvalTracer) -> List[PartialEvalTracer]:
    return t.proto.tracers_in if isinstance(t.proto, InstructionProto) else []


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
    primal_pvals_in, tangent_pvals_in = utils.split_half(pvals_in)

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *utils.split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    Program, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)  # linearize
    primal_pvals, _ = utils.split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    transpose_inputs = consts + [UndefPrimal(p.aval) for p in tangent_pvals_in]
    f_vjp = lambda *cts: eval_program_transposed(Program, transpose_inputs, cts)
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

def eval_program_transposed(
    Program: Program, args: List[Any], cotangents: List[Any]
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

    utils.list_map(write_primal, Program.in_binders, args)
    utils.list_map(write_cotangent, Program.outs, cotangents)
    for eqn in Program.instructions[::-1]:
        primals_in = utils.list_map(read_primal, eqn.inputs)
        cts_in = utils.list_map(read_cotangent, eqn.out_binders)
        
        cts_out = eqn.op.T(cts_in, *primals_in, **eqn.params)
        utils.list_map(write_cotangent, eqn.inputs, cts_out)
    ret = [
        read_cotangent(v)
        for v, x in zip(Program.in_binders, args)
        if type(x) is UndefPrimal
    ]
    return ret


def grad(f):
    def gradfun(x, *xs):
        y, f_vjp = vjp(f, x, *xs)
        if np.shape(y) != ():
            raise TypeError
        out = f_vjp(Array.ones(y.shape, y.dtype))
        return y, out

    return gradfun


class Runtime:
    def __init__(self, root_trace=MainTrace(0, EvalTrace, None)):
        self.trace_stack: List[MainTrace] = []
        self.dynamic_trace: Optional[MainTrace] = None
        self.trace_stack += [root_trace]
        self.node_types = dict()
        self.register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
        self.register_pytree_node(list, lambda l: (None, l), lambda _, xs: list(xs))
        self.register_pytree_node(
            dict,
            lambda d: map(tuple, utils.unzip2(sorted(d.items()))),
            lambda keys, vals: dict(zip(keys, vals)),
        )

    def register_pytree_node(
        self, ty: Type, to_iter: Callable, from_iter: Callable
    ) -> None:
        self.node_types[ty] = NodeType(str(ty), to_iter, from_iter)

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

    def bind(self, op, *args, **params):
        top_trace = self.find_top_trace(args)
        tracers = [self.full_raise(top_trace, arg) for arg in args]
        outs = top_trace.run_op(op, tracers, params)
        lowered = [self.full_lower(out) for out in outs]
        return lowered

    def bind1(self, *args, **params):
        return self.bind(*args, **params)[0]

    def find_top_trace(self, xs) -> Trace:
        top_main = max(
            (
                x._trace.main
                for x in xs
                if isinstance(x, Tracer)
            ),
            default=self.trace_stack[0],
            key=op.attrgetter("level"),
        )
        if self.dynamic_trace and self.dynamic_trace.level > top_main.level:
            top_main = self.dynamic_trace
        return top_main.trace_type(top_main)

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

    def full_lower(self, val: Any):
        if isinstance(val, Tracer):
            return val.full_lower()
        else:
            return val


# def partial_eval_Program(
#     Program: Program,
#     in_unknowns: List[bool],
#     instantiate: Optional[List[bool]] = None,
# ) -> Tuple[Program, Program, List[bool], int]:
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

#     instruction1, instruction2 = [], []
#     map(write, in_unknowns, Program.in_binders)
#     for eqn in Program.instruction:
#         unks_in = map(read, eqn.inputs)
#         if any(unks_in):
#             inputs = [v if unk else new_res(v) for unk, v in zip(unks_in, eqn.inputs)]
#             instruction2.append(Instruction(eqn.op, inputs, eqn.params, eqn.out_binders))
#             map(partial(write, True), eqn.out_binders)
#         else:
#             instruction1.append(eqn)
#             map(partial(write, False), eqn.out_binders)
#     out_unknowns = map(read, Program.outs)
#     if instantiate is not None:
#         for v, uk, inst in zip(Program.outs, out_unknowns, instantiate):
#             if inst and not uk:
#                 new_res(v)
#         out_unknowns = map(op.or_, out_unknowns, instantiate)

#     residuals, num_res = list(residuals), len(residuals)
#     assert all(type(v) is Var for v in residuals), residuals

#     ins1, ins2 = partition_list(in_unknowns, Program.in_binders)
#     outs1, outs2 = partition_list(out_unknowns, Program.outs)

#     Program1 = Program(ins1, instruction1, outs1 + residuals)
#     Program2 = Program(residuals + ins2, instruction2, outs2)
#     typecheck_partial_eval_Program(Program, in_unknowns, out_unknowns, Program1, Program2)

#     return Program1, Program2, out_unknowns, num_res


# def typecheck_partial_eval_Program(Program, unks_in, unks_out, Program1, Program2):
#     Program = typecheck_program(Program)  # (a1,  a2) -> (b1, b2 )
#     Program1ty = typecheck_program(Program1)  #  a1       -> (b1, res)
#     Program2ty = typecheck_program(Program2)  # (res, a2) -> b2

#     a1, a2 = partition_list(unks_in, Program.in_types)
#     b1, b2 = partition_list(unks_out, Program.out_types)
#     b1_, res = split_list(Program1ty.out_types, len(b1))
#     res_, a2_ = split_list(Program2ty.in_types, len(res))
#     b2_ = Program2ty.out_types

#     if Program1ty.in_types != a1:
#         raise TypeError
#     if Program2ty.out_types != b2:
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
# def transpose_Program(
#     Program: Program, undef_primals: Tuple[bool, ...]
# ) -> Tuple[Program, List[Any]]:
#     avals_in, avals_out = typecheck_program(Program)
#     traceable = partial(eval_Program_transposed, Program)
#     args = [UndefPrimal(a) if u else a for a, u in zip(avals_in, undef_primals)]
#     trans_Program, consts, _ = make_Program(traceable, tuple(args), tuple(avals_out))
#     typecheck_program(trans_Program)
#     return trans_Program, consts
