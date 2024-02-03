import slope
import math

XDIMS = (3, 4, 5)

x = slope.arange(math.prod(XDIMS), dtype=slope.float32).reshape(*XDIMS)
x_dot = x.ones_like()
print(f"{x.shape=}")


def f(x):
    return x.sum(-1)


y = f(x)
y_vmap = slope.vmap(f)(x)
y_jit_vmap = slope.jit(slope.vmap(f))(x)
y_vmap_jit = slope.vmap(slope.jit(f))(x)
print(f"{y.shape=}, {y_vmap=} {y_jit_vmap.shape=},  {y_vmap_jit.shape=}")
y_jvp, y_dot_jvp = slope.jvp(f, (x,), (x_dot,))
y_jit_jvp, y_dot_jit_jvp = slope.jvp(slope.jit(f), (x,), (x_dot,))
print(f"{y_jvp.shape=}")
print(f"{y_jit_jvp.shape=}, {y_dot_jit_jvp.shape=}")

loss_fn = lambda *args: f(*args).sum()
(
    L,
    gL_x,
) = slope.value_and_grad(
    loss_fn, argnums=(0,)
)(x)
(
    L_jit,
    gL_x_jit,
) = slope.value_and_grad(
    slope.jit(loss_fn), argnums=(0,)
)(x)
print(f"{L.shape=}, {gL_x.shape=}")
print(f"{L_jit.shape=}, {gL_x_jit.shape=}")
