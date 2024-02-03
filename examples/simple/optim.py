import slope
from slope import nn


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        self.weight = slope.randn((out_dim, in_dim))
        self.bias = slope.zeros(out_dim) if bias else None

    def __call__(self, x):
        x = x @ self.weight.transpose(-1, -2)
        return x + self.bias if self.bias is not None else x


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
optim = nn.SGD(model)
model, optim = train_step(model, (x, y), optim)
print(model.flatten())
print(optim.flatten())
