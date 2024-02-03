import slope

from slope import jit, jvp, grad


def f(x):
    aux = x * 2.0
    y = g(aux)
    return y, aux


def g(x):
    y = x.cos() * 2.0
    return y


def f2(x):
    z = x * 2.0
    y = g(z)
    return y


x = slope.tensor(3.0)
x_dot = slope.tensor(1.0)

y1, aux1 = f(x)
y2, aux2 = jit(f)(x)
(y3, _), aux3 = jvp(f, (x,), (x_dot,), has_aux=True)
(y4, _), aux4 = jvp(jit(f), (x,), (x_dot,), has_aux=True)
print(f"{y1=}; {aux1=}\n" f"{y2=}; {aux2=}\n" f"{y3=}; {aux3=}\n" f"{y4=}; {aux4=}\n")
# jacobian (gradient)
gf_x1, aux_g1 = grad(f, has_aux=True)(x)
gf_x2, aux_g2 = grad(jit(f), has_aux=True)(x)
gf_x3, aux_g3 = jit(grad(jit(f), has_aux=True))(x)
(_, gf_x4), aux_g4 = jvp(f, (x,), (x_dot,), has_aux=True)
(_, gf_x5), aux_g5 = jvp(jit(f), (x,), (x_dot,), has_aux=True)
gf_x6, aux_g6 = jit(grad(f, has_aux=True))(x)
print(
    f"{gf_x1=}; {aux_g1=}\n"
    f"{gf_x2=}; {aux_g2=}\n"
    f"{gf_x3=}; {aux_g3=}\n"
    f"{gf_x4=}; {aux_g4=}\n"
    f"{gf_x5=}; {aux_g5=}\n"
    f"{gf_x6=}; {aux_g6=}\n"
)
# hessian (gradient of gradeint)
ggf_x1, aux_gg1 = grad(grad(f, has_aux=True), has_aux=True)(x)
ggf_x2, aux_gg2 = grad(grad(jit(f), has_aux=True), has_aux=True)(x)
ggf_x3, aux_gg3 = grad(jit(grad(f, has_aux=True)), has_aux=True)(x)
ggf_x4, aux_gg4 = jit(grad(grad(f, has_aux=True), has_aux=True))(x)

(_, ggf_x5), aux_gg5 = jvp(grad(f, has_aux=True), (x,), (x_dot,), has_aux=True)
(_, ggf_x6), aux_gg6 = jvp(jit(grad(f, has_aux=True)), (x,), (x_dot,), has_aux=True)
(_, ggf_x7), aux_gg7 = jvp(jit(grad(f, has_aux=True)), (x,), (x_dot,), has_aux=True)
print(
    f"{ggf_x1=}; {aux_gg1=}\n"
    f"{ggf_x2=}; {aux_gg2=}\n"
    f"{ggf_x3=}; {aux_gg3=}\n"
    f"{ggf_x4=}; {aux_gg4=}\n"
    f"{ggf_x5=}; {aux_gg5=}\n"
    f"{ggf_x6=}; {aux_gg6=}\n"
    f"{ggf_x7=}; {aux_gg7=}\n"
)
