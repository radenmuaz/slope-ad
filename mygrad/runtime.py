from typing import NamedTuple
from contextlib import contextmanager
from typing import Type, Optional, Any, List
import operator as op

from mygrad import arrays, utils, pytrees
import numpy as np


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

    def run_llop(self, LLOp, tracers, params):
        raise NotImplementedError

class Tracer:
    _trace: Trace

    __array_priority__ = 1000

    @property
    def aval(self):
        assert False  # must override

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

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

class EvalTrace(Trace):
    pure = lift = lambda self, x: x
    def run_llop(self, llop, tracers, params):
        return llop.forward(*tracers, **params)

class Runtime:
    JAX_TYPES = {
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
    def __init__ (self, jax_types = JAX_TYPES):
        self.trace_stack: List[MainTrace] = []
        self.dynamic_trace: Optional[MainTrace] = None
        self.node_types = dict()
        self.jax_types = jax_types
        self.trace_stack += [MainTrace(0, EvalTrace, None)]

        self.node_types[tuple] = pytrees.NodeType(
            str(tuple), lambda x: (None, x), lambda _, xs: tuple(xs)
        )
        self.node_types[list] = pytrees.NodeType(
            str(list), lambda x: (None, x), lambda _, xs: list(xs)
        )
        self.node_types[dict] = pytrees.NodeType(
            str(dict),
            lambda d: map(tuple, utils.unzip2(sorted(d.items()))),
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


    def bind(self, prim, *args, **params):
        top_trace = self.find_top_trace(args)
        tracers = [self.full_raise(top_trace, arg) for arg in args]
        outs = top_trace.run_llop(prim, tracers, params)
        return [self.full_lower(out) for out in outs]



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
            assert type(val) in self.jax_types
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


    def get_aval(self, x):
        if isinstance(x, Tracer):
            return x.aval
        elif type(x) in self.jax_types:
            return arrays.ConcreteArray(np.asarray(x))
        else:
            raise TypeError(x)


    @contextmanager
    def new_dynamic(self, main: MainTrace):
        global dynamic_trace
        prev_dynamic_trace, dynamic_trace = dynamic_trace, main
        try:
            yield
        finally:
            dynamic_trace = prev_dynamic_trace

    def zeros_like(self, val):
        aval = self.get_aval(val)
        return np.zeros(aval.shape, aval.dtype)
