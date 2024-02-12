import slope
from slope import nn
import math


class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, act=nn.ReLU()):
        self.flatten_fn = nn.Fn(lambda x: x.reshape(shape=(x.shape[0], math.prod(x.shape[1:]))))
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.act = act

    def __call__(self, x):
        x = self.flatten_fn(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


model = Net(784, 100, 10)


@slope.jit
def train_step(model, batch, optimizer):
    def train_loss_fn(model, batch):
        x, y = batch
        logits = model(x)
        loss = logits.cross_entropy(y) / x.size(0)

        return loss, (model, logits)

    (loss, (model, logits)), gmodel = slope.value_and_grad(train_loss_fn, has_aux=True)(model, batch)
    model, optimizer = optimizer(model, gmodel)
    return loss, logits, model, optimizer, gmodel


optimizer = slope.nn.SGD(model, lr=0.1, momentum=0.9, weight_decay=1e-4)
N = 100
y = slope.arange(10, dtype=slope.float32)[None].expand((N // 10, 10)).reshape(-1)
x = slope.randn(N, 784)
train_step(model, (x, y), optimizer)
with slope.core.Profiling(f"RUN"):
    loss, logits, model, optimizer, gmodel = train_step(model, (x, y), optimizer)
