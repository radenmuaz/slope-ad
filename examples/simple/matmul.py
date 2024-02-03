import slope

XDIMS = (3, 4, 5)
WDIMS = (3, 5, 6)

x = slope.ones(XDIMS)
x_dot = slope.ones(XDIMS)
w = slope.ones(WDIMS)
w_dot = slope.ones(WDIMS)
print(f"{x.shape=}, {w.shape=}")


def f(x, w):
    return x @ w


y = f(x, w)
y_vmap = slope.vmap(f)(x, w)
y_jit_vmap = slope.jit(slope.vmap(f))(x, w)
y_vmap_jit = slope.vmap(slope.jit(f))(x, w)
print(f"{y.shape=}, {y_vmap=} {y_jit_vmap.shape=},  {y_vmap_jit.shape=}")

y_jvp, w_dot_jvp = slope.jvp(f, (x, w), (x_dot, w_dot))
y_jit_jvp, w_dot_jit_jvp = slope.jvp(slope.jit(f), (x, w), (x_dot, w_dot))
print(f"{y_jvp.shape=}, {w_dot_jvp.shape=}")
print(f"{y_jit_jvp.shape=}, {w_dot_jit_jvp.shape=}")

loss_fn = lambda *args: f(*args).sum()
L, (gL_x, gL_w) = slope.value_and_grad(loss_fn, argnums=(0, 1))(x, w)
L_jit, (gL_x_jit, gL_w_jit) = slope.value_and_grad(slope.jit(loss_fn), argnums=(0, 1))(x, w)
print(f"{L.shape=}, {gL_x.shape=}, {gL_w.shape=}")
print(f"{L_jit.shape=}, {gL_x_jit.shape=}, {gL_w_jit.shape=}")
