def vjp_flat(f, *primals_in):
    pvals_in = [PartialVal.known(x) for x in primals_in] + [
        PartialVal.unknown(vspace(get_aval(x))) for x in primals_in
    ]
    primal_pvals_in, tangent_pvals_in = split_half(pvals_in)

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    jaxpr, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)  # linearize
    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    transpose_inputs = consts + [UndefPrimal(p.aval) for p in tangent_pvals_in]
    f_vjp = lambda *cts: eval_jaxpr_transposed(jaxpr, transpose_inputs, cts)
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
    aval: ShapedArray


register_pytree_node(
    UndefPrimal, lambda u: (u.aval, ()), lambda aval, _: UndefPrimal(aval)
)


# NB: the analogous function in JAX is called 'backward_pass'
def eval_jaxpr_transposed(
    jaxpr: Jaxpr, args: List[Any], cotangents: List[Any]
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
            ct_env[x] = add(ct_env[x], val) if x in ct_env else val

    map(write_primal, jaxpr.in_binders, args)
    map(write_cotangent, jaxpr.outs, cotangents)
    for eqn in jaxpr.eqns[::-1]:
        primals_in = map(read_primal, eqn.inputs)
        cts_in = map(read_cotangent, eqn.out_binders)
        rule = transpose_rules[eqn.primitive]
        cts_out = rule(cts_in, *primals_in, **eqn.params)
        map(write_cotangent, eqn.inputs, cts_out)

    return [
        read_cotangent(v)
        for v, x in zip(jaxpr.in_binders, args)
        if type(x) is UndefPrimal
    ]


transpose_rules = {}


def mul_transpose_rule(cts, x, y):
    (z_bar,) = cts
    assert (type(x) is UndefPrimal) ^ (type(y) is UndefPrimal)
    return [mul(z_bar, y), None] if type(x) is UndefPrimal else [None, mul(x, z_bar)]


transpose_rules[mul_p] = mul_transpose_rule


def neg_transpose_rule(cts, x):
    (ybar,) = cts
    assert type(x) is UndefPrimal
    return [neg(ybar)]


transpose_rules[neg_p] = neg_transpose_rule


def add_transpose_rule(cts, x, y):
    (z_bar,) = cts
    return [z_bar, z_bar]


transpose_rules[add_p] = add_transpose_rule


def reduce_sum_transpose_rule(cts, x, *, axis):
    (y_bar,) = cts
    return [broadcast(y_bar, x.aval.shape, axis)]


transpose_rules[reduce_sum_p] = reduce_sum_transpose_rule


def xla_call_transpose_rule(cts, *invals, jaxpr, num_consts):
    del num_consts  # Unused
    undef_primals = [type(x) is UndefPrimal for x in invals]
    transposed_jaxpr, new_consts = transpose_jaxpr(jaxpr, tuple(undef_primals))
    residuals, _ = partition_list(undef_primals, invals)
    outs = bind(
        xla_call_p,
        *new_consts,
        *residuals,
        *cts,
        jaxpr=transposed_jaxpr,
        num_consts=len(new_consts)
    )
    outs = iter(outs)
    return [next(outs) if undef else None for undef in undef_primals]


transpose_rules[xla_call_p] = xla_call_transpose_rule


@lru_cache()
def transpose_jaxpr(
    jaxpr: Jaxpr, undef_primals: Tuple[bool, ...]
) -> Tuple[Jaxpr, List[Any]]:
    avals_in, avals_out = typecheck_jaxpr(jaxpr)
    traceable = partial(eval_jaxpr_transposed, jaxpr)
    args = [UndefPrimal(a) if u else a for a, u in zip(avals_in, undef_primals)]
    trans_jaxpr, consts, _ = make_jaxpr(traceable, tuple(args), tuple(avals_out))
    typecheck_jaxpr(trans_jaxpr)
    return trans_jaxpr, consts


def grad(f):
    def gradfun(x, *xs):
        y, f_vjp = vjp(f, x, *xs)
        if np.shape(y) != ():
            raise TypeError
        x_bar, *_ = f_vjp(np.ones(np.shape(y), np.result_type(y)))
        return x_bar

    return gradfun
