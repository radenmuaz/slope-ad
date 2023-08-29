from slope import rt


@rt.jit
def f(x):
    y = x * 3.0
    z = g(y)
    return z


@rt.jit
def g(x):
    #   return x * rt.array(2.)
    return x * 2.0


#   return x * x

print(rt.grad(f)(rt.array(3.0)))
# print(rt.grad(g)(rt.array(3.)))
# print(g(rt.array(3.)))


# -

# # Here's something of a compositionality stress test:

# # +
# # from core_test.py fun_with_nested_calls_2
# def foo(x):
#   @jit
#   def bar(y):
#     def baz(w):
#       q = jit(lambda x: y)(x)
#       q = q + jit(lambda: y)()
#       q = q + jit(lambda y: w + y)(y)
#       q = jit(lambda w: jit(sin)(x) * y)(1.0) + q
#       return q
#     p, t = jvp(baz, (x + 1.0,), (y,))
#     return t + (x * p)
#   return bar(x)

# def assert_allclose(*vals):
#   for v1, v2 in zip(vals[:-1], vals[1:]):
#     np.testing.assert_allclose(v1, v2)

# ans1 = f(3.)
# ans2 = jit(f)(3.)
# ans3, _ = jvp(f, (3.,), (5.,))
# ans4, _ = jvp(jit(f), (3.,), (5.,))
# assert_allclose(ans1, ans2, ans3, ans4)

# deriv1 = grad(f)(3.)
# deriv2 = grad(jit(f))(3.)
# deriv3 = jit(grad(jit(f)))(3.)
# _, deriv4 = jvp(f, (3.,), (1.,))
# _, deriv5 = jvp(jit(f), (3.,), (1.,))
# assert_allclose(deriv1, deriv2, deriv3, deriv4, deriv5)

# hess1 = grad(grad(f))(3.)
# hess2 = grad(grad(jit(f)))(3.)
# hess3 = grad(jit(grad(f)))(3.)
# hess4 = jit(grad(grad(f)))(3.)
# _, hess5 = jvp(grad(f), (3.,), (1.,))
# _, hess6 = jvp(jit(grad(f)), (3.,), (1.,))
# _, hess7 = jvp(jit(grad(f)), (3.,), (1.,))
# assert_allclose(hess1, hess2, hess3, hess4, hess5, hess6, hess7)