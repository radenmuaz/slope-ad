import slope
from slope import utils
from functools import partial, lru_cache
from typing import Tuple
from slope.dtypes import dtypes
from slope.array import Array
from slope.array_shape import ArrayShape
class Backend:
    def __init__(self, name, default_dtype = dtypes.float32):
        self.name = name
        self.op_impls = dict()
        self.input_handlers = dict(Array=lambda x: x)
        self.default_dtype = default_dtype
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

    def def_compile(self, fn):
        self.compile = fn
    
    def def_op_impl(self, op_name, impl):
        def def_op_impl_(impl):
            self.op_impls[op_name] = impl
        return def_op_impl_
    
    def def_run_op_impl(self, fn):
        self.run_op_impl = fn

    def run_op_impl(self, op_name, *args, **kwargs):
        raise NotImplementedError
        # self.op_impls[op_name](*args, **kwargs)
    
    def def_input_handler(self, typ, fn):
        self.input_handler[typ] = fn

class JitFn:
    def __init__(self, code, fn):
        super().__init__()
        self.code = code
        self.fn = fn

    def __call__(self, *args, **kwargs):
        args = [a.val if isinstance(a, Array) else a for a in args]
        outs = self.fn(*args, **kwargs)
        return [Array(o) o for o in outs]
        # return [Array(o) if isinstance(o, np.ndarray) else o for o in outs]



    # ConvertImpl = BaseOpImpl
    # StopGradientImpl = BaseOpImpl
    # NegImpl = BaseOpImpl
    # ExpImpl = BaseOpImpl
    # LogImpl = BaseOpImpl
    # SqrtImpl = BaseOpImpl

    # AddImpl = BaseOpImpl
    # SubImpl = BaseOpImpl
    # MulImpl = BaseOpImpl
    # DivImpl = BaseOpImpl
    # MaximumImpl = BaseOpImpl
    # EqualImpl = BaseOpImpl
    # NotEqualImpl = BaseOpImpl

    # MaxImpl = BaseOpImpl
    # SumImpl = BaseOpImpl

    # ConstantImpl = BaseOpImpl
    # FullImpl = BaseOpImpl
    # ArangeImpl = BaseOpImpl
    # RandomUniformImpl = BaseOpImpl
    # RandomNormalImpl = BaseOpImpl

    # ReshapeImpl = BaseOpImpl
    # TransposeImpl = BaseOpImpl
    # BroadcastImpl = BaseOpImpl
    # FlipImpl = BaseOpImpl
    # PadImpl = BaseOpImpl
    # ConcatenateImpl = BaseOpImpl
    # SliceImpl = BaseOpImpl
    # GatherImpl = BaseOpImpl
    # ScatterImpl = BaseOpImpl

    
    # stop_gradient = classmethod(lambda cls, x: cls.StopGradientImpl.do(x))
    # convert = classmethod(lambda cls, x, dtype: cls.ConvertImpl.do(x, dtype=dtype))
    # astype = convert
    # neg = classmethod(lambda cls, x: cls.NegImpl.do(x))
    # sqrt = classmethod(lambda cls, x: cls.SqrtImpl.do(x))
    # exp = classmethod(lambda cls, x: cls.ExpImpl.do(x))
    # log = classmethod(lambda cls, x: cls.LogImpl.do(x))

    # add = classmethod(lambda cls, x, other: cls.AddImpl.do(x, other))
    # sub = classmethod(lambda cls, x, other: cls.SubImpl.do(x, other))
    # mul = classmethod(lambda cls, x, other: cls.MulImpl.do(x, other))
    # div = classmethod(lambda cls, x, other: cls.DivImpl.do(x, other))
    # maximum = classmethod(lambda cls, x, other: cls.MaximumImpl.do(x, other))
    # equal = classmethod(lambda cls, x, other: cls.EqualImpl.do(x, other))
    # not_equal = classmethod(lambda cls, x, other: cls.NotEqualImpl.do(x, other))

    # max = classmethod(lambda cls, x, other: cls.MaxImpl.do(x, other))
    # sum = classmethod(lambda cls, x, other: cls.SumImpl.do(x, other))

    # constant = classmethod(
    #     lambda cls, val, dtype: cls.ConstantImpl.do(val=val, dtype=dtype)
    # )
    # full = classmethod(
    #     lambda cls, val, dtype: cls.FullImpl.do(val=val, dtype=dtype)
    # )
    # arange = classmethod(lambda cls, start, stop, stride, dtype: cls.ArangeImpl.do(start, stop, stride, dtype))
    # random_uniform = classmethod(lambda cls, x, other: cls.RandomUniform.do(x, other))
    # rand = random_uniform
    # random_normal = classmethod(lambda cls, x, other: cls.RandomNormal.do(x, other))
    # randn = random_normal
    # broadcast = classmethod(
    #     lambda cls, x, shape, axes: cls.BroadcastImpl.do(x, shape=shape, axes=axes)
    # )
    # reshape = classmethod(lambda cls, x, shape: cls.ReshapeImpl.do(x, shape=shape))
    # transpose = classmethod(lambda cls, x, axes: cls.TransposeImpl.do(x, axes=axes))
    # flip = classmethod(lambda cls, x, axes: cls.FlipImpl.do(x, axes=axes))
    # pad = classmethod(lambda cls, x, lo, hi, interior, value: cls.PadImpl.do(x, lo, hi, interior, value))
    # slice = classmethod(lambda cls, x, axes: cls.SliceImpl.do(x, axes=axes))
    # concatenate = classmethod(lambda cls, x, axes: cls.ConcatenateImpl.do(x, axes=axes))
    # gather = classmethod(lambda cls, x, axes: cls.GatherImpl.do(x, axes=axes))
    # scatter = classmethod(lambda cls, x, axes: cls.ScatterImpl.do(x, axes=axes))


# class BaseBuilder:
#     pass

# class BaseConst:
#     pass


# class BaseParam:
#     pass
