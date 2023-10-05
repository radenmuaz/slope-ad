import slope
from slope import nn
from slope import environment as sev
import math

def f(x, y):
    return x + y

x = sev.full((1,), 1)
y = sev.full((1,), 2)

xs = (x, (x, x))
ys = (y, (y, y))
res = slope.tree_map(f, xs, ys)
print(res)


# class Linear(slope.core.Module):
#     def __init__(self, in_dim, out_dim, bias=False):
#         self.weight = sev.randn((out_dim, in_dim))
#         self.bias = sev.zeros(out_dim) if bias else None

#     def __call__(self, x):
#         x = x.dot(self.weight.T())
#         return x + self.bias if self.bias is not None else x

# x = sev.ones((1, 2))
# y = sev.full((1,), 1)

# model = Linear(2, 3, 1)
# g_model = Linear(2, 3, 1)

# sgd = nn.SGD(model)

# @slope.jit
# def loss_fn(model, batch):
#     x, y = batch
#     y_hat = model(x)
#     loss = (y - y_hat).sum()
#     return loss


# print(loss_fn(model, (x, y)))
# g_loss_fn = slope.grad(loss_fn)
# print(g_loss_fn(model, (x, y)).flatten())
