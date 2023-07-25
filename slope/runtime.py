from slope.ad import Trace, MainTrace, NodeType, EvalTrace
from slope.tracer_array import TracerArray
from typing import Callable, Type, Any, List, Optional
from contextlib import contextmanager
from slope import utils
import operator
from slope.numpy_backend import numpy_backend


def register_pytree_node(ty: Type, to_iter: Callable, from_iter: Callable) -> None:
    node_types[ty] = NodeType(str(ty), to_iter, from_iter)


@contextmanager
def new_main(trace_type: Type["Trace"], global_data=None):
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()


@contextmanager
def new_dynamic(main: MainTrace):
    prev_dynamic_trace, dynamic_trace = dynamic_trace, main
    try:
        yield
    finally:
        dynamic_trace = prev_dynamic_trace


def bind(op, *args, **params):
    top_trace = find_top_trace(args)
    tracers = [full_raise(top_trace, arg) for arg in args]
    outs = top_trace.run_op(op, tracers, params)
    lowered = [full_lower(out) for out in outs]
    return lowered


def bind1(*args, **params):
    return bind(*args, **params)[0]


def find_top_trace(xs) -> Trace:
    top_main = max(
        (x._trace.main for x in xs if isinstance(x, TracerArray)),
        default=trace_stack[0],
        key=operator.attrgetter("level"),
    )
    if dynamic_trace and dynamic_trace.level > top_main.level:
        top_main = dynamic_trace
    return top_main.trace_type(top_main)


def full_raise(trace: Trace, val: Any) -> TracerArray:
    if not isinstance(val, TracerArray):
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


def full_lower(val: Any):
    if isinstance(val, TracerArray):
        return val.full_lower()
    else:
        return val


backend = numpy_backend
trace_stack: List[MainTrace] = [MainTrace(0, EvalTrace, None)]
dynamic_trace: Optional[MainTrace] = None
node_types = dict()
register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
register_pytree_node(list, lambda l: (None, l), lambda _, xs: list(xs))
register_pytree_node(
    dict,
    lambda d: map(tuple, utils.unzip2(sorted(d.items()))),
    lambda keys, vals: dict(zip(keys, vals)),
)
