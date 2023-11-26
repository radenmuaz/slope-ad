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
print(f"{y1=}; {aux1=}\n"
      f"{y2=}; {aux2=}\n"
      f"{y3=}; {aux3=}\n"
      f"{y4=}; {aux4=}\n")
# jacobian (gradient)
grad_f_x1, aux_grad_1 = grad(f, has_aux=True)(x)
grad_f_x2, aux_grad_2 = grad(jit(f), has_aux=True)(x)
grad_f_x3, aux_grad_3 = jit(grad(jit(f), has_aux=True))(x)
(_, grad_f_x4), aux_grad_4 = jvp(f, (x,), (x_dot,), has_aux=True)
(_, grad_f_x5), aux_grad_5 = jvp(jit(f), (x,), (x_dot,), has_aux=True)
grad_f_x6, aux_grad_6 = jit(grad(f, has_aux=True))(x)
print(f"{grad_f_x1=}; {aux_grad_1=}\n"
      f"{grad_f_x2=}; {aux_grad_2=}\n"
      f"{grad_f_x3=}; {aux_grad_3=}\n"
      f"{grad_f_x4=}; {aux_grad_4=}\n"
      f"{grad_f_x5=}; {aux_grad_5=}\n"
      f"{grad_f_x6=}; {aux_grad_6=}\n"
      )
# hessian (gradient of gradeint)
grad_grad_f_x1, aux_grad_grad_1 = grad(grad(f, has_aux=True), has_aux=True)(x)
grad_grad_f_x2, aux_grad_grad_2  = grad(grad(jit(f), has_aux=True), has_aux=True)(x)
grad_grad_f_x3, aux_grad_grad_3  = grad(jit(grad(f, has_aux=True)), has_aux=True)(x)
grad_grad_f_x4, aux_grad_grad_4  = jit(grad(grad(f, has_aux=True), has_aux=True))(x)

(_, grad_grad_f_x5), aux_grad_grad_5  = jvp(grad(f, has_aux=True), (x,), (x_dot,), has_aux=True)
(_, grad_grad_f_x6), aux_grad_grad_6  = jvp(jit(grad(f, has_aux=True)), (x,), (x_dot,), has_aux=True)
(_, grad_grad_f_x7), aux_grad_grad_7  = jvp(jit(grad(f, has_aux=True)), (x,), (x_dot,), has_aux=True)
print(f"{grad_grad_f_x1=}; {aux_grad_grad_1=}\n"
      f"{grad_grad_f_x2=}; {aux_grad_grad_2=}\n"
      f"{grad_grad_f_x3=}; {aux_grad_grad_3=}\n"
      f"{grad_grad_f_x4=}; {aux_grad_grad_4=}\n"
      f"{grad_grad_f_x5=}; {aux_grad_grad_5=}\n"
      f"{grad_grad_f_x6=}; {aux_grad_grad_6=}\n"
      f"{grad_grad_f_x7=}; {aux_grad_grad_7=}\n"

      )