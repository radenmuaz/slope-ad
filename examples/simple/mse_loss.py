import slope
import slope.nn as nn
import math

x = slope.ones((1, 2))
y = slope.full((1,), 1)
model = nn.Linear(2, 1)


@slope.jit
def loss_fn(model, batch):
    x, y = batch
    y_hat = model(x)
    loss = (y - y_hat).pow(2).sum()
    return loss


L = loss_fn(model, (x, y))
gloss_fn = slope.grad(loss_fn)
gL = gloss_fn(model, (x, y)).flatten()

print(f"{L=}, {gL=}")
