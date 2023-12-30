import slope
from slope import nn

import math

# def f(x, y):
#     return x + y

# x = slope.full((1,), 1)
# y = slope.full((1,), 2)

# xs = (x, (x, x))
# ys = (y, (y, y))
# res = slope.tree_map(f, xs, ys)
# print(res)


class Linear(slope.core.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        self.weight = slope.randn((out_dim, in_dim))
        self.bias = slope.zeros(out_dim) if bias else None

    def __call__(self, x):
        x = x.dot(self.weight.T())
        return x + self.bias if self.bias is not None else x

# x = slope.ones((1, 2))
# y = slope.full((1,), 1)

# model = Linear(2, 3, 1)
# g_model = Linear(2, 3, 1)

# sgd = nn.SGD(model)
# print(sgd.iters)
# model, sgd = sgd(model, g_model)
# print(sgd.iters)


def loss_fn(model, batch):
    x, y = batch
    y_hat = model(x)
    loss = (y - y_hat).sum()
    return loss

g_loss_fn = slope.grad(loss_fn)

@slope.jit
def train_step(model, batch, optim):
    g_model = g_loss_fn(model, batch)
    model, optim = optim(model, g_model)
    return model, optim


x = slope.ones((1, 2))
y = slope.ones((1, 1))
model = Linear(2, 3, 1)
sgd = nn.SGD(model)
# print(loss_fn(model, (x, y)))
# print(g_loss_fn(model, (x, y)).flatten())
model, sgd = train_step(model, (x, y), sgd)
print(model.flatten())
print(sgd.flatten())
# g_loss_fn = slope.grad(loss_fn)
