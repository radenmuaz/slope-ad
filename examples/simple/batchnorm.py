import slope
import slope.nn as nn
import math


class Net(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim):
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.bn_out = nn.BatchNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def __call__(self, x, training=False):
        x = self.in_proj(x).relu()
        x = self.bn_out(x, training=training)
        x = self.out_proj(x)
        return (x, self) if training else x


def loss_fn(model, batch):
    x, y = batch
    y_hat, model = model(x, training=True)
    loss = (y - y_hat).pow(2).sum()
    return loss, model


vgloss_fn = slope.value_and_grad(loss_fn, has_aux=True)


@slope.jit
def train_step(model, batch, optimizer):
    ((loss, model), gloss_model) = vgloss_fn(model, batch)
    model, optimizer = optimizer(model, gloss_model)
    return loss, model, optimizer


x = slope.ones((3, 2))
y = slope.ones(3, 1)
batch = (x, y)
model = Net(2, 5, 1)
optim = nn.SGD(model, lr=1e-3, momentum=0.8, weight_decay=1e-5)
print(f"Before: {model.bn_out.running_mean=}")
loss, model, optimizer_ = train_step(model, batch, optim)
print(f"After: {model.bn_out.running_mean=}")
