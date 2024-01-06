import slope

from slope import jit, jvp, grad

def f(x):
    z = x * 2.0
    y = g(z)
    return y

def g(x):
    z = x.cos()
    y = z * 2.0
    return y

x = slope.tensor(3.0)
x_dot = slope.tensor(1.0)

y1 = f(x)
y2 = jit(f)(x)
y3, _ = jvp(f, (x,), (x_dot,))
y4, _ = jvp(jit(f), (x,), (x_dot,))
print(y1, y2, y3, y4)

# jacobian (gradient)
gf_x1 = grad(f)(x)
gf_x2 = grad(jit(f))(x)
gf_x3 = jit(grad(jit(f)))(x)
_, gf_x4 = jvp(f, (x,), (x_dot,))
_, gf_x5 = jvp(jit(f), (x,), (x_dot,))
gf_x6 = jit(grad(f))(x)
print(gf_x1, gf_x2, gf_x3, gf_x4, gf_x5, gf_x6)

# hessian (gradient of gradeint)
ggf_x1 = grad(grad(f))(x)
ggf_x2 = grad(grad(jit(f)))(x)
ggf_x3 = grad(jit(grad(f)))(x)
ggf_x4 = jit(grad(grad(f)))(x)
print(ggf_x1, ggf_x2, ggf_x3, ggf_x4)

_, ggf_x5 = jvp(grad(f), (x,), (x_dot,))
_, ggf_x6 = jvp(jit(grad(f)), (x,), (x_dot,))
_, ggf_x7 = jvp(jit(grad(f)), (x,), (x_dot,))
print(ggf_x5, ggf_x6, ggf_x7)
