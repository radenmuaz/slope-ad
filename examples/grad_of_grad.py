import slope

from slope import jit, jvp, grad


def f(x):
    y = x * 2.0
    z = g(y)
    return z


def g(x):
    return x.cos() * 2.0


x = slope.tensor(3.0)
x_dot = slope.tensor(1.0)

ans1 = f(x)  # ; print(f"{ans1=}")
ans2 = jit(f)(x)  # BUG: jit cache not cleared when jvp jit ;# print(f"{ans2=}")
ans3, _ = jvp(f, (x,), (x_dot,))  # ; print(f"{ans3=}")
ans4, _ = jvp(jit(f), (x,), (x_dot,))  # ; print(f"{ans4=}")
print(ans1, ans2, ans3, ans4)

deriv1 = grad(f)(x)  # ; print(f"{deriv1=}")
deriv2 = grad(jit(f))(x)  # ; print(f"{deriv2=}")
deriv3 = jit(grad(jit(f)))(x)  # ;print(f"{deriv3=}")
_, deriv4 = jvp(f, (x,), (x_dot,))  # ;print(f"{deriv4=}")
_, deriv5 = jvp(jit(f), (x,), (x_dot,))  # ;print(f"{deriv5=}")
deriv6 = jit(grad(f))(x)  # ; print(f"{deriv6=}")
print(deriv1, deriv2, deriv3, deriv4, deriv5, deriv6)

hess1 = grad(grad(f))(x)
hess2 = grad(grad(jit(f)))(x)
hess3 = grad(jit(grad(f)))(x)
hess4 = jit(grad(grad(f)))(x)
print(hess1, hess2, hess3, hess4)

_, hess5 = jvp(grad(f), (x,), (x_dot,))
_, hess6 = jvp(jit(grad(f)), (x,), (x_dot,))
_, hess7 = jvp(jit(grad(f)), (x,), (x_dot,))
print(hess5, hess6, hess7)
