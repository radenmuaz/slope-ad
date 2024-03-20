import slope
from slope import nn
import math


def train_loss_fn(w1, w2, x):
    x = x @ w1
    # x = x.exp()
    # x = x.transpose(0,1) @ w2
    loss = x.sum()
    return loss


N = 100
x = slope.randn(N, 784)
w1 = slope.randn(784, 10)
w2 = slope.randn(100, 10)

fn = slope.jit(train_loss_fn)
with slope.core.Timing(f"RUN: "):
    fn(w1, w2, x)
grad_fn = slope.jit(slope.value_and_grad(train_loss_fn))
grad_fn(w1, w2, x)
print(grad_fn.lower(w1, w2, x).program)
# with slope.core.Timing(f"GRAD: "): grad_fn(w1, w2, x)
with slope.core.Profiling(f"GRAD"):
    grad_fn(w1, w2, x)
# slope.jit(lambda w, x: slope.jvp(train_loss_fn, (w, x), (w.ones_like(), x.ones_like())))(w, x)
# with slope.core.Profiling(f"RUN"):
