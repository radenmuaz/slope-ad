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


xla_call_p = Op("xla_call")


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
    return partial(execute_compiled, compiled, [v.aval for v in jaxpr.outs])


def _xla_consts(c: xe.XlaBuilder, consts: List[Any]) -> List[xe.XlaOp]:
    unique_consts = {id(cnst): cnst for cnst in consts}
    xla_consts = {
        id_: xops.ConstantLiteral(c, cnst) for id_, cnst in unique_consts.items()
    }
    return [xla_consts[id(cnst)] for cnst in consts]


def _xla_params(c: xe.XlaBuilder, avals_in: List[TensorShape]) -> List[xe.XlaOp]:
    return [xops.Parameter(c, i, _xla_shape(a)) for i, a in enumerate(avals_in)]


def _xla_shape(aval: TensorShape) -> xe.Shape:
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
        rule = xla_translations[eqn.Op]
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


def handle_result(aval: TensorShape, buf):
    del aval  # Unused for now
    return np.asarray(buf)


xla_translations = {}


def direct_translation(op, c, in_avals, in_vals):
    del c, in_avals
    return [op(*in_vals)]


xla_translations[add_p] = partial(direct_translation, xops.Add)
xla_translations[mul_p] = partial(direct_translation, xops.Mul)
xla_translations[neg_p] = partial(direct_translation, xops.Neg)
xla_translations[sin_p] = partial(direct_translation, xops.Sin)
xla_translations[cos_p] = partial(direct_translation, xops.Cos)
xla_translations[greater_p] = partial(direct_translation, xops.Gt)
xla_translations[less_p] = partial(direct_translation, xops.Lt)


def reduce_sum_translation(c, in_avals, in_vals, *, axis):
    (x_aval,), (x,) = in_avals, in_vals
    zero = xops.ConstantLiteral(c, np.array(0, x_aval.dtype))
    subc = xc.XlaBuilder("add")
    shape = _xla_shape(TensorShape((), x_aval.dtype))
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
    axis_size: int, batch_dim: BatchAxis, aval: TensorShape
) -> TensorShape:
    if batch_dim is not_mapped:
        return aval
    else:
        shape = list(aval.shape)
        shape.insemygrad.RT(batch_dim, axis_size)
        return TensorShape(tuple(shape), aval.dtype)


def xla_call_shape_eval_rule(*in_types, jaxpr, num_consts):
    del num_consts  # Unused
    jaxpr_type = typecheck_jaxpr(jaxpr)
    if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
        raise TypeError
    return jaxpr_type.out_types


shape_eval_rules[xla_call_p] = xla_call_shape_eval_rule


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


def handle_result(aval: TensorShape, buf):  # noqa: F811
    return DeviceArray(aval, buf)


class DeviceArray:
    buf: Any
    aval: TensorShape

    def __init__(self, aval, buf):
        self.aval = aval
        self.buf = buf

    dtype = property(lambda self: self.aval.dtype)
    shape = property(lambda self: self.aval.shape)
    ndim = property(lambda self: self.aval.ndim)

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
        pp(eqn.Op.name)
        >> pp_params(params_without_jaxpr)
        >> pp(
            " ".join(names[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs)
        )
    )
    return vcat([lhs >> pp(" = ") >> rhs, pp_jaxpr(eqn.params["jaxpr"]).indent(2)])


pp_rules[xla_call_p] = pprint_xla_call
