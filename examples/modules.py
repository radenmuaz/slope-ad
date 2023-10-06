import slope
from slope import environment as sev
import math


class Linear(slope.core.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        self.weight = slope.randn((out_dim, in_dim))
        self.bias = slope.zeros(out_dim) if bias else None

    def __call__(self, x):
        x = x.dot(self.weight.T())
        return x + self.bias if self.bias is not None else x


class MLP(slope.core.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        self.t = slope.randn(1)
        self.linear1 = Linear(in_dim, hid_dim)
        self.linear2 = Linear(hid_dim, out_dim)

    def __call__(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x


x = slope.ones((1, 2))
y = slope.full((1,), 1)

# # model = Linear(2, 1)
# model = MLP(2, 3, 1)


# @slope.jit
# def loss_fn(model, batch):
#     x, y = batch
#     y_hat = model(x)
#     loss = (y - y_hat).sum()
#     return loss


# # print(loss_fn(model, (x, y)))
# g_loss_fn = slope.grad(loss_fn)
# print(g_loss_fn(model, (x, y)).flatten())



# print(g_loss_fn(model, (x, y)).flatten()[1])
# breakpoint()
# print(slope.jit(g_loss_fn)(model, (x, y)).flatten()[1])

# g_loss_fn = slope.jit(g_loss_fn)
# print(loss_fn(model, (x, y)))


# a = (y,(y,y))
# af, at = slope.tree_flatten(a)
# b = slope.tree_unflatten(at, af)
# print(a)
# print(b)

# print(f"{af=}")
# print(f"{ta=}")


m = Linear(2, 1, True)
m = MLP(2, 3, 1)
mf, mt = slope.tree_flatten(m)
m_hat = slope.tree_unflatten(mt, mf)
print(f"{mf=}")
print(f"{mt=}")
