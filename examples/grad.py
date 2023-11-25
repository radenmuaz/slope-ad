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
grad_f_x1 = grad(f)(x)
grad_f_x2 = grad(jit(f))(x)
grad_f_x3 = jit(grad(jit(f)))(x)
_, grad_f_x4 = jvp(f, (x,), (x_dot,))
_, grad_f_x5 = jvp(jit(f), (x,), (x_dot,))
grad_f_x6 = jit(grad(f))(x)
print(grad_f_x1, grad_f_x2, grad_f_x3, grad_f_x4, grad_f_x5, grad_f_x6)

# hessian (gradient of gradeint)
grad_grad_f_x1 = grad(grad(f))(x)
grad_grad_f_x2 = grad(grad(jit(f)))(x)
grad_grad_f_x3 = grad(jit(grad(f)))(x)
grad_grad_f_x4 = jit(grad(grad(f)))(x)
print(grad_grad_f_x1, grad_grad_f_x2, grad_grad_f_x3, grad_grad_f_x4)

_, grad_grad_f_x5 = jvp(grad(f), (x,), (x_dot,))
_, grad_grad_f_x6 = jvp(jit(grad(f)), (x,), (x_dot,))
_, grad_grad_f_x7 = jvp(jit(grad(f)), (x,), (x_dot,))
print(grad_grad_f_x5, grad_grad_f_x6, grad_grad_f_x7)
