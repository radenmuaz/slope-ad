import slope
from slope import nn
import math
def train_loss_fn(w, x):
    logits = x @ w
    loss = logits.sum()
    return loss

N = 100
x = slope.randn(N, 784)
w = slope.randn(784, 10)
slope.jit(train_loss_fn)(w, x)
slope.jit(lambda w, x: slope.jvp(train_loss_fn, (w, x), (w.ones_like(), x.ones_like())))(w, x)
slope.jit(slope.value_and_grad(train_loss_fn))(w, x)
# with slope.core.Profiling(f"RUN"):
