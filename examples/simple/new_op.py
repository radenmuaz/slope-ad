import slope

@slope.core.backend.procedure_set.register()
def relu(x):
    w = slope.zeros_like(x)
    y = x.maximum(w)
    return y
relu = slope.core.Operator.unary("relu")
slope.M().backend.operator_set.register(relu)

@relu.set_method
def jvp(self, primals, tangents):
    def _balanced_eq(x, z, y):
        xz = (x == z).where(slope.ones_like(z), slope.zeros_like(z))
        yz = (y == z).where(slope.full_like(z, 2), slope.ones_like(z))
        return xz / yz

    (x,), (x_dot,) = primals, tangents
    y = x.relu()
    w = slope.zeros_like(x)
    w_dot  = slope.ones_like(x)
    y_dot = x_dot * _balanced_eq(x, y, w) + w_dot * _balanced_eq(w, y, x)
    return [y], [y_dot]

@relu.set_method
def T(self, cotangents, x):
    (gL_y,) = cotangents
    return [gL_y, None]

@slope.jit
def f(x):
    y = x.relu()
    # y = y + 10
    y = y.sum()

    return y

x1 = slope.tensor([1., 2., -1., 0.])
print(f"{x1=}")
y1 = f(x1)
print(f"{y1=}")

x1_dot = slope.ones_like(x1)
y1, y1_dot = slope.jvp(f, (x1,), (x1_dot,))
print(f"{y1=}, {y1_dot=}")

gf_x1 = slope.grad(f)(x1)
print(gf_x1)
