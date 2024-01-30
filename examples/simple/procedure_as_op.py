import slope

# procedures are basically functions accessible by Tensor.procedure syntax
@slope.core.backend.procedure_set.register()
def my_relu(x):
    w = slope.zeros_like(x)
    y = x.maximum(w)
    return y

# To override a procedure gradient rule, define a new operator with the same name as procedure
# the impl will default to tracing procedure as a Program,
# not need to define backend-specific impl code.
@slope.backend.operator_set.register("my_relu")
class MyReLU(slope.core.UnaryOperator):
    def jvp(self, primals, tangents):
        def _balanced_eq(x, z, y):
            xz = (x == z).where(slope.ones_like(z), slope.zeros_like(z))
            yz = (y == z).where(slope.full_like(z, 2), slope.ones_like(z))
            return xz / yz

        (x,), (x_dot,) = primals, tangents
        y = x.my_relu()
        w = slope.zeros_like(x)
        w_dot  = slope.ones_like(x)
        y_dot = x_dot * _balanced_eq(x, y, w) + w_dot * _balanced_eq(w, y, x)
        return [y], [y_dot]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        return [gL_y, None]

@slope.jit
def f(x):
    y = x.my_relu()
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
