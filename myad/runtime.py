from typing import NamedTuple
from contextlib import contextmanager
from typing import Type, Optional, Any, List, Tuple, Callable
import operator as op
import itertools
import numpy as np

from myad import utils
from myad.tracing.base import Trace, Tracer, MainTrace
from myad.tracing.eager import EagerEvalTrace
from myad.tracing.ir import Jaxpr, JaxprBuilder, JaxprTrace
from myad.arrays import ShapedArray

from myad.pytrees import NodeType, PyTreeDef
from myad import pytrees

from functools import lru_cache


class Runtime:
    RTs = []

    @classmethod
    @property
    def active(cls, *args, **kwargs):
        if len(cls.RTs) == 0:
            print("init new runtime")
            cls(*args, **kwargs)
        return cls.RTs[-1]

    def __init__(self, root_trace=MainTrace(0, EagerEvalTrace, None)):
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
            lambda d: map(tuple, utils.unzip2(sorted(d.items()))),
            lambda keys, vals: dict(zip(keys, vals)),
        )
        self.RTs += [self]

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

        # raise TypeError(x)

    @lru_cache()
    def make_jaxpr(
        self,
        f: Callable,
        *avals_in: ShapedArray,
    ) -> Tuple[Jaxpr, List[Any], PyTreeDef]:
        avals_in, in_tree = pytrees.tree_flatten(avals_in)
        f, out_tree = pytrees.flatten_fun(f, in_tree)

        builder = JaxprBuilder()
        with self.new_main(JaxprTrace, builder) as main:
            with self.new_dynamic(main):
                trace = JaxprTrace(main)
                tracers_in = [trace.new_arg(aval) for aval in avals_in]
                outs = f(*tracers_in)
                tracers_out = [self.full_raise(trace, out) for out in outs]
                jaxpr, consts = builder.build(tracers_in, tracers_out)
        return jaxpr, consts, out_tree()
