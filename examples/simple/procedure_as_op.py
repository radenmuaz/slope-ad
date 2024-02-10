import slope


# procedures are basically functions accessible by Tensor.procedure syntax
@slope.core.backend.procedure_set.register()
def my_relu(x):
    y = x.maximum(slope.zeros_like(x))
    return y


# To override a procedure gradient rule, define a new operator with the same name as procedure
# the impl will default to tracing procedure as a Program if not defined.

@slope.backend.operator_set.register("my_relu")
class ReLUOp(slope.core.UnaryOperator):
    def jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        y = x.my_relu()
        y_dot = x_dot * (x == y).where(slope.ones_like(x), slope.zeros_like(x))
        return [y], [y_dot]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        gL_x = (x.my_relu() > x.zeros_like()).cast(gL_y.dtype) * gL_y
        return [gL_x]

@slope.jit
def f(x):
    y = x.my_relu()
    y = y.sum()

    return y


x1 = slope.tensor([1.0, 2.0, -1.0, 0.0])
print(f"{x1=}")
y1 = f(x1)
print(f"{y1=}")

x1_dot = slope.ones_like(x1)
y1, y1_dot = slope.jvp(f, (x1,), (x1_dot,))
print(f"{y1=}, {y1_dot=}")

gf_x1 = slope.grad(f)(x1)
print(gf_x1)
