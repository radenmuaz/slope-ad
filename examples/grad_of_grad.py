import slope
from slope import numpy as snp
from slope import jit, jvp, grad
# @jit
def f(x):
  y = x * 2.
  z = g(y)
  return z

# @jit
def g(x):
  return x.cos() * 2.


# print(f(3.))
# print(grad(g)(snp.array(3.)))


# x = snp.array(1)
# gg = slope.grad(slope.jit(g))
# for i in range(3):
#     print(gg(snp.array(float(i + 1))))
#     print(slope.machine.backend.callable.cache_info())
#     print()

# def assert_allclose(*vals):
#   for v1, v2 in zip(vals[:-1], vals[1:]):
#     np.testing.assert_allclose(v1, v2)

# ans1 = f(3.)
# ans2 = jit(f)(3.)
# ans3, _ = jvp(f, (3.,), (5.,))
# ans4, _ = jvp(jit(f), (3.,), (5.,))
# print(ans1, ans2, ans3, ans4)

# deriv1 = grad(f)(3.)
# deriv2 = grad(jit(f))(3.)
# deriv3 = jit(grad(jit(f)))(3.)
# _, deriv4 = jvp(f, (3.,), (1.,))
# _, deriv5 = jvp(jit(f), (3.,), (1.,))
# print(deriv1, deriv2, deriv3, deriv4, deriv5)
x = snp.array(3.)
x_dot = snp.array(1.)
hess1 = grad(grad(f))(x)
hess2 = grad(grad(jit(f)))(x)
hess3 = grad(jit(grad(f)))(x)
hess4 = jit(grad(grad(f)))(x)
_, hess5 = jvp(grad(f), (x,), (x_dot,))
_, hess6 = jvp(jit(grad(f)), (x,), (x_dot,))
_, hess7 = jvp(jit(grad(f)), (x,), (x_dot,))
print(hess1, hess2, hess3, hess4, hess5, hess6, hess7)

# # from core_test.py fun_with_nested_calls_2
# def foo(x):
#   @jit
#   def bar(y):
#     def baz(w):
#       q = jit(lambda x: y)(x)
#       q = q + jit(lambda: y)()
#       q = q + jit(lambda y: w + y)(y)
#       q = jit(lambda w: jit(x.sin()) * y)(1.0) + q
#       return q
#     p, t = jvp(baz, (x + 1.0,), (y,))
#     return t + (x * p)
#   return bar(x)