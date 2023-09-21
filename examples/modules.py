import slope
from slope import environment as sev
import math


@slope.core.as_module
class Linear:
    def __init__(self, in_dim, out_dim, bias=False):
        self.weight = sev.randn((out_dim, in_dim))
        self.bias = sev.zeros(out_dim) if bias else None

    def __call__(self, x):
        x = x.dot(self.weight.T())
        return x + self.bias if self.bias is not None else x


@slope.core.as_module
class MLP:
    def __init__(self, in_dim, hid_dim, out_dim):
        self.linear1 = Linear(in_dim, hid_dim)
        self.linear2 = Linear(hid_dim, out_dim)

    def __call__(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x


# model = Linear(2, 1)
model = MLP(2, 3, 1)


# model = model.unflatten(*model.flatten())
# @slope.jit
def loss_fn(model, batch):
    x, y = batch
    y_hat = model(x)
    # loss = y_hat.sum()
    loss = (y - y_hat).sum()
    return loss


x = sev.ones((1, 2))
y = sev.ones(1)
# print(loss_fn(model, (x, y)))
# print(slope.jit(loss_fn)(model, (x, y)))


g_loss_fn = slope.grad(loss_fn)
# print(g_loss_fn(model, (x, y)).flatten())
# print(g_loss_fn(model, (x, y)).flatten()[1])
# breakpoint()
print(slope.jit(g_loss_fn)(model, (x, y)).flatten()[1])

# g_loss_fn = slope.jit(g_loss_fn)
# print(loss_fn(model, (x, y)))
