import slope
from slope import utils
from functools import partial, lru_cache
from typing import Tuple
from slope.base_array import Array
from functools import partial


class Backend:
    def __init__(self, name, default_dtype=Array.float32):
        self.name = name
        self.input_handlers = dict(Array=lambda x: x)
        self.default_dtype = default_dtype
        self.impls = dict()
        self.callable = lru_cache(self.callable)

    def callable(
        self,
        hashable_prog: utils.IDHashable,
        hashable_consts: Tuple[utils.IDHashable, ...],
    ):
        prog: slope.ad.Prog = hashable_prog.val
        slope.ad.typecheck_prog(prog)
        consts = [x.val for x in hashable_consts]
        in_avals = [v.aval for v in prog.in_binders[len(consts) :]]
        compiled = self.compile(
            prog, consts, in_avals, name=f"{self.__class__.__name__.lower()}_fn"
        )
        return compiled

    def execute_compiled(self, compiled, out_avals, *args):
        input_bufs = [self.input_handlers[type(x)](x) for x in args]
        out_bufs = compiled.execute(input_bufs)
        return [Array(buf) for aval, buf in zip(out_avals, out_bufs)]

    def compile(self, prog, consts, in_avals, name: str):
        raise NotImplementedError

    def set_compile(self, fn):
        self.compile = partial(fn, self)

    def set_impl(self, name):
        def set_impl_(fn):
            self.impls[name] = fn

        return set_impl_

    def set_run_impl(self, fn):
        self.run_impl = partial(fn, self)

    def run_impl(self, op_name, *args, **kwargs):
        raise NotImplementedError
        # self.op_impls[op_name](*args, **kwargs)

    def set_input_handler(self, typ, fn):
        self.input_handlers[typ] = fn

    # def __getattr__(self, attr):
    #     if attr in self.impls.keys():
    #         return partial(self.run_impl, self, attr)
    #     print(f"{attr} not found in {self}.op_impls")
    #     breakpoint()
    #     return getattr(self.attr)


class JitFn:
    def __init__(self, code, fn):
        super().__init__()
        self.code = code
        self.fn = fn

    def __call__(self, *args, **kwargs):
        args = [a.val if isinstance(a, Array) else a for a in args]
        outs = self.fn(*args, **kwargs)
        return [Array(o) for o in outs]
        # return [Array(o) if isinstance(o, np.ndarray) else o for o in outs]
