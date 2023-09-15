import slope
from slope import environment as sev
import math

@slope.core.as_module
class Linear:
    def __init__(self, in_dim, out_dim, bias=True):
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
        x = self.linear2(x)
        return x


# model = Linear(2, 1)
model = MLP(2, 3, 1)


def loss_fn(model, batch):
    x, y = batch
    y_hat = model(x)
    loss = (y - y_hat).sum()
    return loss


g_loss_fn = slope.grad(loss_fn)
x = sev.ones((1, 2))
y = sev.ones(1)

print(loss_fn(model, (x, y)))
g_loss = g_loss_fn(model, (x, y))
print(g_loss.state_dict)
