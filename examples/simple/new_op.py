import slope


relu = slope.core.Operator.unary("relu", is_procedure=True)
slope.M().backend.operator_set.register(relu)

@relu.set_method
def procedure(self, x):
    w = slope.zeros_like(x)
    y = x.maximum(w)
    return [y]

@relu.set_method
def jvp(self, primals, tangents):
    def _balanced_eq(x, z, y):
        xz = (x == z).where(slope.ones_like(z), slope.zeros_like(z))
        yz = (y == z).where(slope.full_like(z, 2), slope.ones_like(z))
        return xz / yz

    (x), (x_dot,) = primals, tangents
    w = slope.zeros_like(x)
    y = x.maximum(w)
    y_dot = x_dot * _balanced_eq(x, y, w) + w_dot * _balanced_eq(w, y, x)
    return [y], [y_dot]


@relu.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [grad_L_y, None]

x1 = slope.tensor([1., 2., -1., 0.])
print(f"{x1=}")
y1 = x1.relu()
print(f"{y1=}")