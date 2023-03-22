import myad


class LLOp:
    @staticmethod
    def forward(*args):
        raise NotImplementedError

    @staticmethod
    def vmap(*args):
        raise NotImplementedError

    @staticmethod
    def jvp(*args):
        raise NotImplementedError

    @staticmethod
    def shape_forward(*args):
        raise NotImplementedError

    @classmethod
    def bind(cls, *args, **params):
        top_trace = myad.RT.find_top_trace(args)
        tracers = [myad.RT.full_raise(top_trace, arg) for arg in args]
        outs = top_trace.run_llop(cls, tracers, params)
        lowered = [myad.RT.full_lower(out) for out in outs]
        return lowered

    @classmethod
    def bind1(cls, *args, **params):
        top_trace = myad.RT.find_top_trace(args)
        tracers = [myad.RT.full_raise(top_trace, arg) for arg in args]
        outs = top_trace.run_llop(cls, tracers, params)
        lowered = [myad.RT.full_lower(out) for out in outs]
        return lowered[0]