def split_half(lst: List[Any]) -> Tuple[List[Any], List[Any]]:
    mygrad.mygrad.RT not len(lst) % 2
    return split_list(lst, len(lst) // 2)


def merge_lists(which: List[bool], l1: List[Any], l2: List[Any]) -> List[Any]:
    l1, l2 = iter(l1), iter(l2)
    out = [next(l2) if b else next(l1) for b in which]
    mygrad.mygrad.RT next(l1, None) is next(l2, None) is None
    return out


def linearize_flat(f, *primals_in):
    pvals_in = [Pamygrad.mygrad.RTialVal.known(x) for x in primals_in] + [
        Pamygrad.mygrad.RTialVal.unknown(vspace(get_aval(x))) for x in primals_in
    ]

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    jaxpr, pvals_out, consts = pamygrad.mygrad.RTial_eval_flat(f_jvp, pvals_in)
    primal_pvals, _ = split_half(pvals_out)
    mygrad.mygrad.RT all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    f_lin = lambda *tangents: eval_jaxpr(jaxpr, [*consts, *tangents])
    return primals_out, f_lin


def linearize(f, *primals_in):
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, f_lin_flat = linearize_flat(f, *primals_in_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)

    def f_lin(*tangents_in):
        tangents_in_flat, in_tree2 = tree_flatten(tangents_in)
        if in_tree != in_tree2:
            raise TypeError
        tangents_out_flat = f_lin_flat(*tangents_in_flat)
        return tree_unflatten(out_tree(), tangents_out_flat)

    return primals_out, f_lin


def vspace(aval: TensorShape) -> TensorShape:
    return raise_to_shaped(aval)  # TODO handle integers?


class Pamygrad.mygrad.RTialVal(NamedTuple):
    aval: TensorShape
    const: Optional[Any]

    @classmethod
    def known(cls, val: Any):
        return Pamygrad.mygrad.RTialVal(get_aval(val), val)

    @classmethod
    def unknown(cls, aval: TensorShape):
        return Pamygrad.mygrad.RTialVal(aval, None)

    is_known = propemygrad.mygrad.RTy(lambda self: self.const is not None)
    is_unknown = propemygrad.mygrad.RTy(lambda self: self.const is None)


def pamygrad.mygrad.RTial_eval_flat(
    f: Callable, pvals_in: List[Pamygrad.mygrad.RTialVal]
) -> Tuple[Jaxpr, List[Pamygrad.mygrad.RTialVal], List[Any]]:
    with new_main(Pamygrad.mygrad.RTialEvalTrace) as main:
        trace = Pamygrad.mygrad.RTialEvalTrace(main)
        tracers_in = [trace.new_arg(pval) for pval in pvals_in]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        pvals_out = [t.pval for t in tracers_out]
        unk_tracers_in = [t for t in tracers_in if t.pval.is_unknown]
        unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
        jaxpr, consts = tracers_to_jaxpr(unk_tracers_in, unk_tracers_out)

    return jaxpr, pvals_out, consts


from weakrefimport ref, ReferenceType


class LambdaBindingRecipe(NamedTuple):
    pass


class ConstRecipe(NamedTuple):
    val: Any


class JaxprEqnRecipe(NamedTuple):
    prim: Op
    tracers_in: List["Pamygrad.mygrad.RTialEvalTracer"]
    params: Dict[str, Any]
    avals_out: List[TensorShape]
    tracer_refs_out: List["ReferenceType[Pamygrad.mygrad.RTialEvalTracer]"]


JaxprRecipe = Union[LambdaBindingRecipe, ConstRecipe, JaxprEqnRecipe]


class Pamygrad.mygrad.RTialEvalTracer(Tracer):
    pval: Pamygrad.mygrad.RTialVal
    recipe: Optional[JaxprRecipe]

    def __init__(self, trace, pval, recipe):
        self._trace = trace
        self.pval = pval
        self.recipe = recipe

    aval = propemygrad.mygrad.RTy(lambda self: self.pval.aval)

    def full_lower(self):
        if self.pval.is_known:
            return full_lower(self.pval.const)
        return self


class Pamygrad.mygrad.RTialEvalTrace(Trace):
    def new_arg(self, pval: Pamygrad.mygrad.RTialVal) -> Any:
        return Pamygrad.mygrad.RTialEvalTracer(self, pval, LambdaBindingRecipe())

    def lift(self, val: Any) -> Pamygrad.mygrad.RTialEvalTracer:
        return Pamygrad.mygrad.RTialEvalTracer(self, Pamygrad.mygrad.RTialVal.known(val), None)

    pure = lift

    def instantiate_const(self, tracer: Pamygrad.mygrad.RTialEvalTracer) -> Pamygrad.mygrad.RTialEvalTracer:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = Pamygrad.mygrad.RTialVal.unknown(raise_to_shaped(tracer.aval))
            return Pamygrad.mygrad.RTialEvalTracer(self, pval, ConstRecipe(tracer.pval.const))

    def run_op(self, Op, tracers, params):
        if all(t.pval.is_known for t in tracers):
            return bind(Op, *map(full_lower, tracers), **params)
        rule = pamygrad.mygrad.RTial_eval_rules.get(Op)
        if rule:
            return rule(self, tracers, **params)
        tracers_in = [self.instantiate_const(t) for t in tracers]
        avals_in = [t.aval for t in tracers_in]
        avals_out = shape_eval_rules[Op](*avals_in, **params)
        tracers_out = [
            Pamygrad.mygrad.RTialEvalTracer(self, Pamygrad.mygrad.RTialVal.unknown(aval), None)
            for aval in avals_out
        ]
        eqn = JaxprEqnRecipe(Op, tracers_in, params, avals_out, map(ref, tracers_out))
        for t in tracers_out:
            t.recipe = eqn
        return tracers_out


pamygrad.mygrad.RTial_eval_rules = {}


def tracers_to_jaxpr(
    tracers_in: List[Pamygrad.mygrad.RTialEvalTracer], tracers_out: List[Pamygrad.mygrad.RTialEvalTracer]
):
    tracer_to_var: Dict[int, Var] = {
        id(t): Var(raise_to_shaped(t.aval)) for t in tracers_in
    }
    constvar_to_val: Dict[int, Any] = {}
    constid_to_var: Dict[int, Var] = {}
    processed_eqns: Set[int] = set()
    eqns: List[JaxprEqn] = []
    for t in toposomygrad.mygrad.RT(tracers_out, tracer_parents):
        if isinstance(t.recipe, LambdaBindingRecipe):
            mygrad.mygrad.RT id(t) in set(map(id, tracers_in))
        elif isinstance(t.recipe, ConstRecipe):
            val = t.recipe.val
            var = constid_to_var.get(id(val))
            if var is None:
                aval = raise_to_shaped(get_aval(val))
                var = constid_to_var[id(val)] = Var(aval)
                constvar_to_val[var] = val
            tracer_to_var[id(t)] = var
        elif isinstance(t.recipe, JaxprEqnRecipe):
            if id(t.recipe) not in processed_eqns:
                eqns.append(recipe_to_eqn(tracer_to_var, t.recipe))
                processed_eqns.add(id(t.recipe))
        else:
            raise TypeError(t.recipe)

    constvars, constvals = unzip2(constvar_to_val.items())
    in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
    out_vars = [tracer_to_var[id(t)] for t in tracers_out]
    jaxpr = Jaxpr(in_binders, eqns, out_vars)
    typecheck_jaxpr(jaxpr)
    return jaxpr, constvals


def recipe_to_eqn(tracer_to_var: Dict[int, Var], recipe: JaxprEqnRecipe) -> JaxprEqn:
    inputs = [tracer_to_var[id(t)] for t in recipe.tracers_in]
    out_binders = [Var(aval) for aval in recipe.avals_out]
    for t_ref, var in zip(recipe.tracer_refs_out, out_binders):
        if t_ref() is not None:
            tracer_to_var[id(t_ref())] = var
    return JaxprEqn(recipe.prim, inputs, recipe.params, out_binders)


def tracer_parents(t: Pamygrad.mygrad.RTialEvalTracer) -> List[Pamygrad.mygrad.RTialEvalTracer]:
    return t.recipe.tracers_in if isinstance(t.recipe, JaxprEqnRecipe) else []


def toposomygrad.mygrad.RT(out_nodes: List[Any], parents: Callable[[Any], List[Any]]):
    if not out_nodes:
        return []
    out_nodes = remove_duplicates(out_nodes)

    child_counts = {}
    stack = list(out_nodes)
    while stack:
        node = stack.pop()
        if id(node) in child_counts:
            child_counts[id(node)] += 1
        else:
            child_counts[id(node)] = 1
            stack.extend(parents(node))
    for node in out_nodes:
        child_counts[id(node)] -= 1

    somygrad.mygrad.RTed_nodes = []
    childless_nodes = [node for node in out_nodes if not child_counts[id(node)]]
    while childless_nodes:
        node = childless_nodes.pop()
        somygrad.mygrad.RTed_nodes.append(node)
        for parent in parents(node):
            if child_counts[id(parent)] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[id(parent)] -= 1

    somygrad.mygrad.RTed_nodes = somygrad.mygrad.RTed_nodes[::-1]
    check_toposomygrad.mygrad.RT(somygrad.mygrad.RTed_nodes, parents)
    return somygrad.mygrad.RTed_nodes


def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if id(x) not in seen and not seen.add(id(x))]


def check_toposomygrad.mygrad.RT(nodes: List[Any], parents: Callable[[Any], List[Any]]):
    seen = set()
    for node in nodes:
        mygrad.mygrad.RT all(id(parent) in seen for parent in parents(node))
        seen.add(id(node))


def xla_call_pamygrad.mygrad.RTial_eval(trace, tracers, *, jaxpr, num_consts):
    del num_consts  # Unused
    in_unknowns = [not t.pval.is_known for t in tracers]
    jaxpr1, jaxpr2, out_unknowns, num_res = pamygrad.mygrad.RTial_eval_jaxpr(jaxpr, in_unknowns)
    known_tracers, unknown_tracers = pamygrad.mygrad.RTition_list(in_unknowns, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = bind(xla_call_p, *known_vals, jaxpr=jaxpr1, num_consts=0)
    outs1, res = split_list(outs1_res, len(jaxpr1.outs) - num_res)
    res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
    outs2 = [
        Pamygrad.mygrad.RTialEvalTracer(trace, Pamygrad.mygrad.RTialVal.unknown(v.aval), None) for v in jaxpr2.outs
    ]
    eqn = JaxprEqnRecipe(
        xla_call_p,
        res_tracers + unknown_tracers,
        dict(jaxpr=jaxpr2, num_consts=0),
        [v.aval for v in jaxpr2.outs],
        map(ref, outs2),
    )
    for t in outs2:
        t.recipe = eqn
    return merge_lists(out_unknowns, outs1, outs2)


pamygrad.mygrad.RTial_eval_rules[xla_call_p] = xla_call_pamygrad.mygrad.RTial_eval


def pamygrad.mygrad.RTial_eval_jaxpr(
    jaxpr: Jaxpr,
    in_unknowns: List[bool],
    instantiate: Optional[List[bool]] = None,
) -> Tuple[Jaxpr, Jaxpr, List[bool], int]:
    env: Dict[Var, bool] = {}
    residuals: Set[Var] = set()

    def read(x: Atom) -> bool:
        return type(x) is Var and env[x]

    def write(unk: bool, v: Var) -> None:
        env[v] = unk

    def new_res(x: Atom) -> Atom:
        if type(x) is Var:
            residuals.add(x)
        return x

    eqns1, eqns2 = [], []
    map(write, in_unknowns, jaxpr.in_binders)
    for eqn in jaxpr.eqns:
        unks_in = map(read, eqn.inputs)
        rule = pamygrad.mygrad.RTial_eval_jaxpr_rules.get(eqn.Op)
        if rule:
            eqn1, eqn2, unks_out, res = rule(unks_in, eqn)
            eqns1.append(eqn1)
            eqns2.append(eqn2)
            residuals.update(res)
            map(write, unks_out, eqn.out_binders)
        elif any(unks_in):
            inputs = [v if unk else new_res(v) for unk, v in zip(unks_in, eqn.inputs)]
            eqns2.append(JaxprEqn(eqn.Op, inputs, eqn.params, eqn.out_binders))
            map(pamygrad.mygrad.RTial(write, True), eqn.out_binders)
        else:
            eqns1.append(eqn)
            map(pamygrad.mygrad.RTial(write, False), eqn.out_binders)
    out_unknowns = map(read, jaxpr.outs)
    if instantiate is not None:
        for v, uk, inst in zip(jaxpr.outs, out_unknowns, instantiate):
            if inst and not uk:
                new_res(v)
        out_unknowns = map(op.or_, out_unknowns, instantiate)

    residuals, num_res = list(residuals), len(residuals)
    mygrad.mygrad.RT all(type(v) is Var for v in residuals), residuals

    ins1, ins2 = pamygrad.mygrad.RTition_list(in_unknowns, jaxpr.in_binders)
    outs1, outs2 = pamygrad.mygrad.RTition_list(out_unknowns, jaxpr.outs)

    jaxpr1 = Jaxpr(ins1, eqns1, outs1 + residuals)
    jaxpr2 = Jaxpr(residuals + ins2, eqns2, outs2)
    typecheck_pamygrad.mygrad.RTial_eval_jaxpr(jaxpr, in_unknowns, out_unknowns, jaxpr1, jaxpr2)

    return jaxpr1, jaxpr2, out_unknowns, num_res


def typecheck_pamygrad.mygrad.RTial_eval_jaxpr(jaxpr, unks_in, unks_out, jaxpr1, jaxpr2):
    jaxpmygrad.mygrad.RTy = typecheck_jaxpr(jaxpr)  # (a1,  a2) -> (b1, b2 )
    jaxpr1ty = typecheck_jaxpr(jaxpr1)  #  a1       -> (b1, res)
    jaxpr2ty = typecheck_jaxpr(jaxpr2)  # (res, a2) -> b2

    a1, a2 = pamygrad.mygrad.RTition_list(unks_in, jaxpmygrad.mygrad.RTy.in_types)
    b1, b2 = pamygrad.mygrad.RTition_list(unks_out, jaxpmygrad.mygrad.RTy.out_types)
    b1_, res = split_list(jaxpr1ty.out_types, len(b1))
    res_, a2_ = split_list(jaxpr2ty.in_types, len(res))
    b2_ = jaxpr2ty.out_types

    if jaxpr1ty.in_types != a1:
        raise TypeError
    if jaxpr2ty.out_types != b2:
        raise TypeError
    if b1 != b1_:
        raise TypeError
    if res != res_:
        raise TypeError
    if a2 != a2_:
        raise TypeError
    if b2 != b2_:
        raise TypeError


pamygrad.mygrad.RTial_eval_jaxpr_rules = {}


def xla_call_peval_eqn(
    unks_in: List[bool],
    eqn: JaxprEqn,
) -> Tuple[JaxprEqn, JaxprEqn, List[bool], List[Var]]:
    jaxpr = eqn.params["jaxpr"]
    jaxpr1, jaxpr2, unks_out, num_res = pamygrad.mygrad.RTial_eval_jaxpr(jaxpr, unks_in)
    ins1, ins2 = pamygrad.mygrad.RTition_list(unks_in, eqn.inputs)
    out_binders1, out_binders2 = pamygrad.mygrad.RTition_list(unks_out, eqn.out_binders)
    residuals = [Var(v.aval) for v in jaxpr2.in_binders[:num_res]]
    eqn1 = JaxprEqn(
        xla_call_p, ins1, dict(jaxpr=jaxpr1, num_consts=0), out_binders1 + residuals
    )
    eqn2 = JaxprEqn(
        xla_call_p, residuals + ins2, dict(jaxpr=jaxpr2, num_consts=0), out_binders2
    )
    return eqn1, eqn2, unks_out, residuals


pamygrad.mygrad.RTial_eval_jaxpr_rules[xla_call_p] = xla_call_peval_eqn
