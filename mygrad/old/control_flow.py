def cond(pred, true_fn, false_fn, *operands):
    avals_in = [raise_to_shaped(get_aval(x)) for x in operands]
    true_jaxpr, true_consts, out_tree = make_jaxpr(true_fn, *avals_in)
    false_jaxpr, false_consts, out_tree_ = make_jaxpr(false_fn, *avals_in)
    if out_tree != out_tree_:
        raise TypeError
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    if typecheck_jaxpr(true_jaxpr) != typecheck_jaxpr(false_jaxpr):
        raise TypeError
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *operands,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr
    )
    return tree_unflatten(out_tree, outs)


cond_p = LLOp("cond")


def _join_jaxpr_consts(
    jaxpr1: Jaxpr, jaxpr2: Jaxpr, n1: int, n2: int
) -> Tuple[Jaxpr, Jaxpr]:
    jaxpr1_type, jaxpr2_type = typecheck_jaxpr(jaxpr1), typecheck_jaxpr(jaxpr2)
    assert jaxpr1_type.in_types[n1:] == jaxpr2_type.in_types[n2:]
    consts1, rest1 = split_list(jaxpr1.in_binders, n1)
    consts2, rest2 = split_list(jaxpr2.in_binders, n2)
    new_jaxpr1 = Jaxpr(consts1 + consts2 + rest1, jaxpr1.eqns, jaxpr1.outs)
    new_jaxpr2 = Jaxpr(consts1 + consts2 + rest2, jaxpr2.eqns, jaxpr2.outs)
    return new_jaxpr1, new_jaxpr2


def bind_cond(pred, *args, true_jaxpr, false_jaxpr):
    assert len(args) == len(true_jaxpr.in_binders) == len(false_jaxpr.in_binders)
    return bind(cond_p, pred, *args, true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)


def cond_impl(pred, *operands, true_jaxpr, false_jaxpr):
    if pred:
        return eval_jaxpr(true_jaxpr, operands)
    else:
        return eval_jaxpr(false_jaxpr, operands)


impl_rules[cond_p] = cond_impl


def cond_jvp_rule(primals, tangents, *, true_jaxpr, false_jaxpr):
    pred, *primals = primals
    _, *tangents = tangents
    true_jaxpr, true_consts = jvp_jaxpr(true_jaxpr)
    false_jaxpr, false_consts = jvp_jaxpr(false_jaxpr)
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    assert typecheck_jaxpr(true_jaxpr) == typecheck_jaxpr(false_jaxpr)
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *primals,
        *tangents,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr
    )
    primals_out, tangents_out = split_half(outs)
    return primals_out, tangents_out


jvp_rules[cond_p] = cond_jvp_rule


def cond_vmap_rule(axis_size, vals_in, dims_in, *, true_jaxpr, false_jaxpr):
    pred, *vals_in = vals_in
    pred_dim, *dims_in = dims_in
    if pred_dim is not not_mapped:
        raise NotImplementedError  # TODO
    true_jaxpr, true_consts = vmap_jaxpr(true_jaxpr, axis_size, tuple(dims_in))
    false_jaxpr, false_consts = vmap_jaxpr(false_jaxpr, axis_size, tuple(dims_in))
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    assert typecheck_jaxpr(true_jaxpr) == typecheck_jaxpr(false_jaxpr)
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *vals_in,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr
    )
    return outs, [0] * len(outs)


vmap_rules[cond_p] = cond_vmap_rule


def cond_abstract_eval(pred_type, *in_types, true_jaxpr, false_jaxpr):
    if pred_type != ShapedArray((), np.dtype("bool")):
        raise TypeError
    jaxpr_type = typecheck_jaxpr(true_jaxpr)
    if jaxpr_type != typecheck_jaxpr(false_jaxpr):
        raise TypeError
    if not all(t1 == t2 for t1, t2 in zip(jaxpr_type.in_types, in_types)):
        raise TypeError
    return jaxpr_type.out_types


abstract_eval_rules[cond_p] = cond_abstract_eval


def cond_translation(c, in_avals, in_vals, *, true_jaxpr, false_jaxpr):
    del in_avals  # Unused
    pred, *in_vals = in_vals
    flat_vals, in_tree = tree_flatten(in_vals)
    operand = xops.Tuple(c, flat_vals)
    operand_shape = c.get_shape(operand)

    def make_comp(name: str, jaxpr: Jaxpr) -> xe.XlaComputation:
        c = xc.XlaBuilder(name)
        operand = xops.Parameter(c, 0, operand_shape)
        operands = tree_unflatten(in_tree, destructure_tuple(c, operand))
        outs = jaxpr_subcomp(c, jaxpr, operands)
        return c.build(xops.Tuple(c, outs))

    true_comp = make_comp("true_fn", true_jaxpr)
    false_comp = make_comp("false_fn", false_jaxpr)

    int_etype = xc.dtype_to_etype(np.dtype("int32"))
    out = xops.Conditional(
        xops.ConvertElementType(pred, int_etype), [false_comp, true_comp], [operand] * 2
    )
    return destructure_tuple(c, out)


xla_translations[cond_p] = cond_translation


def cond_partial_eval(trace, tracers, *, true_jaxpr, false_jaxpr):
    pred_tracer, *tracers = tracers
    assert pred_tracer.pval.is_known
    pred = pred_tracer.pval.const
    in_uks = [not t.pval.is_known for t in tracers]

    *jaxprs, out_uks, num_res = _cond_partial_eval(true_jaxpr, false_jaxpr, in_uks)
    t_jaxpr1, f_jaxpr1, t_jaxpr2, f_jaxpr2 = jaxprs

    known_tracers, unknown_tracers = partition_list(in_uks, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = bind_cond(pred, *known_vals, true_jaxpr=t_jaxpr1, false_jaxpr=f_jaxpr1)
    outs1, res = split_list(outs1_res, len(outs1_res) - num_res)
    pred_tracer_ = trace.instantiate_const(full_raise(trace, pred_tracer))
    res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
    outs2 = [
        PartialEvalTracer(trace, PartialVal.unknown(v.aval), None)
        for v in t_jaxpr2.outs
    ]
    eqn = JaxprEqnRecipe(
        cond_p,
        [pred_tracer_, *res_tracers, *unknown_tracers],
        dict(true_jaxpr=t_jaxpr2, false_jaxpr=f_jaxpr2),
        [v.aval for v in t_jaxpr2.outs],
        map(ref, outs2),
    )
    for t in outs2:
        t.recipe = eqn
    return merge_lists(out_uks, outs1, outs2)


partial_eval_rules[cond_p] = cond_partial_eval


def _cond_partial_eval(
    true_jaxpr: Jaxpr, false_jaxpr: Jaxpr, in_uks: List[bool]
) -> Tuple[Jaxpr, Jaxpr, Jaxpr, Jaxpr, List[bool], int]:
    _, _, t_out_uks, _ = partial_eval_jaxpr(true_jaxpr, in_uks)
    _, _, f_out_uks, _ = partial_eval_jaxpr(false_jaxpr, in_uks)
    out_uks = map(op.or_, t_out_uks, f_out_uks)

    t_jaxpr1, t_jaxpr2, _, t_nres = partial_eval_jaxpr(true_jaxpr, in_uks, out_uks)
    f_jaxpr1, f_jaxpr2, _, f_nres = partial_eval_jaxpr(false_jaxpr, in_uks, out_uks)

    t_jaxpr1, f_jaxpr1 = _join_jaxpr_res(t_jaxpr1, f_jaxpr1, t_nres, f_nres)
    t_jaxpr2, f_jaxpr2 = _join_jaxpr_consts(t_jaxpr2, f_jaxpr2, t_nres, f_nres)
    assert typecheck_jaxpr(t_jaxpr1) == typecheck_jaxpr(f_jaxpr1)
    assert typecheck_jaxpr(t_jaxpr2) == typecheck_jaxpr(f_jaxpr2)
    num_res = t_nres + f_nres

    return t_jaxpr1, f_jaxpr1, t_jaxpr2, f_jaxpr2, out_uks, num_res


def _join_jaxpr_res(
    jaxpr1: Jaxpr, jaxpr2: Jaxpr, n1: int, n2: int
) -> Tuple[Jaxpr, Jaxpr]:
    jaxpr1_type, jaxpr2_type = typecheck_jaxpr(jaxpr1), typecheck_jaxpr(jaxpr2)
    out_types1, _ = split_list(jaxpr1_type.out_types, len(jaxpr1.outs) - n1)
    out_types2, _ = split_list(jaxpr2_type.out_types, len(jaxpr2.outs) - n2)
    assert out_types1 == out_types2
    outs1, res1 = split_list(jaxpr1.outs, len(jaxpr1.outs) - n1)
    outs2, res2 = split_list(jaxpr2.outs, len(jaxpr2.outs) - n2)
    zeros_like1 = [Lit(np.zeros(v.aval.shape, v.aval.dtype)) for v in res1]
    zeros_like2 = [Lit(np.zeros(v.aval.shape, v.aval.dtype)) for v in res2]
    new_jaxpr1 = Jaxpr(jaxpr1.in_binders, jaxpr1.eqns, outs1 + res1 + zeros_like2)
    new_jaxpr2 = Jaxpr(jaxpr2.in_binders, jaxpr2.eqns, outs2 + zeros_like1 + res2)
    return new_jaxpr1, new_jaxpr2


def cond_peval_eqn(
    unks_in: List[bool],
    eqn: JaxprEqn,
) -> Tuple[JaxprEqn, JaxprEqn, List[bool], List[Atom]]:
    pred_unk, *unks_in = unks_in
    assert not pred_unk
    true_jaxpr, false_jaxpr = eqn.params["true_jaxpr"], eqn.params["false_jaxpr"]
    *jaxprs, unks_out, num_res = _cond_partial_eval(true_jaxpr, false_jaxpr, unks_in)
    t_jaxpr1, f_jaxpr1, t_jaxpr2, f_jaxpr2 = jaxprs
    ins1, ins2 = partition_list(unks_in, eqn.inputs[1:])
    outs1, outs2 = partition_list(unks_out, eqn.out_binders)
    residuals, _ = split_list(t_jaxpr2.in_binders, num_res)
    eqn1 = JaxprEqn(
        cond_p,
        [eqn.inputs[0], *ins1],
        dict(true_jaxpr=t_jaxpr1, false_jaxpr=f_jaxpr1),
        outs1 + residuals,
    )
    eqn2 = JaxprEqn(
        cond_p,
        [eqn.inputs[0], *residuals, *ins2],
        dict(true_jaxpr=t_jaxpr2, false_jaxpr=f_jaxpr2),
        outs2,
    )
    res = [eqn.inputs[0], *residuals] if type(eqn.inputs[0]) is Var else residuals
    return eqn1, eqn2, unks_out, res


partial_eval_jaxpr_rules[cond_p] = cond_peval_eqn


def cond_transpose_rule(cts, pred, *invals, true_jaxpr, false_jaxpr):
    undef_primals = tuple(type(x) is UndefPrimal for x in invals)
    true_jaxpr, true_consts = transpose_jaxpr(true_jaxpr, undef_primals)
    false_jaxpr, false_consts = transpose_jaxpr(false_jaxpr, undef_primals)
    true_jaxpr, false_jaxpr = _join_jaxpr_consts(
        true_jaxpr, false_jaxpr, len(true_consts), len(false_consts)
    )
    res = [x for x in invals if type(x) is not UndefPrimal]
    outs = bind_cond(
        pred,
        *true_consts,
        *false_consts,
        *res,
        *cts,
        true_jaxpr=true_jaxpr,
        false_jaxpr=false_jaxpr
    )
    outs = iter(outs)
    return [None] + [next(outs) if type(x) is UndefPrimal else None for x in invals]


transpose_rules[cond_p] = cond_transpose_rule


def pprint_cond(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
    true_jaxpr, false_jaxpr = eqn.params["true_jaxpr"], eqn.params["false_jaxpr"]
    new_params = {k: v for k, v in eqn.params.items() if not k.endswith("jaxpr")}
    lhs = pp(" ".join(var_str(names, v) for v in eqn.out_binders))
    rhs = (
        pp(eqn.LLOp.name)
        >> pp_params(new_params)
        >> pp(
            " ".join(names[x] if isinstance(x, Var) else str(x.val) for x in eqn.inputs)
        )
    )
    return vcat(
        [
            lhs >> pp(" = ") >> rhs,
            pp_jaxpr(true_jaxpr).indent(2),
            pp_jaxpr(false_jaxpr).indent(2),
        ]
    )


pp_rules[cond_p] = pprint_cond
