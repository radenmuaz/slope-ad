from typing import NamedTuple
from contextlib import contextmanager
from typing import Type, Optional, Any, List
import operator as op

from mygrad import arrays, reg
import numpy as np


class MainTrace(NamedTuple):
    level: int
    trace_type: Type["Trace"]
    global_data: Optional[Any]


@contextmanager
def new_main(trace_type: Type["Trace"], global_data=None):
    level = len(reg.trace_stack)
    main = MainTrace(level, trace_type, global_data)
    reg.trace_stack.append(main)

    try:
        yield main
    finally:
        reg.trace_stack.pop()


class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val):
        assert False  # must override

    def lift(self, val):
        assert False  # must override

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


class EvalTrace(trc.Trace):
    pure = lift = lambda self, x: x  # no boxing in Tracers needed

    def process_primitive(self, primitive, tracers, params):
        return reg.impl_rules[primitive](*tracers, **params)


def find_top_trace(xs) -> Trace:
    top_main = max(
        (x._trace.main for x in xs if isinstance(x, Tracer)),
        default=reg.trace_stack[0],
        key=op.attrgetter("level"),
    )
    if reg.dynamic_trace and reg.dynamic_trace.level > top_main.level:
        top_main = reg.dynamic_trace
    return top_main.trace_type(top_main)


def full_lower(val: Any):
    if isinstance(val, Tracer):
        return val.full_lower()
    else:
        return val


def full_raise(trace: Trace, val: Any) -> Tracer:
    if not isinstance(val, Tracer):
        assert type(val) in reg.jax_types
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


def get_aval(x):
    if isinstance(x, Tracer):
        return x.aval
    elif type(x) in reg.jax_types:
        return arrays.ConcreteArray(np.asarray(x))
    else:
        raise TypeError(x)


@contextmanager
def new_dynamic(main: MainTrace):
    global dynamic_trace
    prev_dynamic_trace, dynamic_trace = dynamic_trace, main
    try:
        yield
    finally:
        dynamic_trace = prev_dynamic_trace


def zeros_like(val):
    aval = get_aval(val)
    return np.zeros(aval.shape, aval.dtype)
