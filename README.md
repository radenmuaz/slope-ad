```
   /|s
  / |l
 /  |o
/___|p 
 slope
```

Slope is a small automatic differentation (AD) engine, focused on machine learning (ML).
Currently it uses IREE and StableHLO MLIR as backend.
It is written to be educational and hackable, yet does things end-to-end, from training to deployment.
- Tensor API like Pytorch, 
- Higher-order derivatives, stateless functional API and pytrees like JAX
- Operator decompositions and easily add new backend like tinygrad.

Slope aims to be some sort of ML *middleware* -- you can go up and use Slope as building blocks to write NN modules, classes, trainer, self-attention, etc. yet you can go down to define new operators, backends and extend the core in `src/slope/{core.py, operators.py, procedures.py, nn.py, backends}`

# Install

NOTE: For Mac user you must use Python 3.11 because it is minimum version for latest IREE wheel for Mac.

Install dependencies, clone this repo then install editable
```
pip install iree-compiler iree-runtime
https://github.com/radenmuaz/slope
pip install -e .
```

Or you can just copy `src/slope` to your projects.

# Quickstart

Tensor operation semantics feel like pytorch, but AD semantics like JAX (or functorch)

Check out examples in `examples/simple` for short snippets.
MNIST classifier example is in `example/mnist_mlp.py`

`src/slope/nn.py` provides nn.Module system like in Pytorch, but with usage API like JAX
You are encouraged to read the source, starting from `slope/__init__.py`, `slope/core.py` and `slope/backends/iree.py`

# Slope has familiar Pytorch-like syntax

Most of the things familiar in Pytorch works in Slope, probably.

```python
import slope
x = slope.ones(2, 5)
w = slope.arange(15).reshape(5,3)
b = slope.tensor([1., 2., 3.], dtype=slope.float32)
y = x @ w
print(y)
```

# Every operations are compiled with slope.jit


Actually when these lines are run, each operation calls jitted as individual programs eagerly.

```python

x = slope.full((1,), 2.)
x = x + 1
```

To prevent eager jit, write code in function and use slope.jit.
Then call the function
```python
@slope.jit
def f(x):
    y = x * x
    return y
# Alternative way to jit
# f = slope.jit(f)

x = slope.full((1,), 2.)
y = f(x)
```

TODO: Besides eager and jit, make async eager like Pytorch.

# Reveal contents of jitted function

A decorated function with `slope.jit` is an instance object of `slope.core.jit`,
It has several utility methods for printing the generated backend code and exporting the function

```python
f_jitobj = f.get_jit_object(x)
print(f_jitobj.code)
f_jitobj.export('./f_export_folder') # see the folder of what are exported.
```

# Derivatives and gradients

Slope has several AD functions, like `slope.jvp` `slope.vjp` and `slope.grad`

To do the usual backprop things:
```python
def model_foward(x, w):
    y = x @ w
    return y

def loss_fn(y_hat, y):
    return ((y_hat - y)**2).sum()
gloss_fn = slope.grad(loss_fn, argnums=(0, 1))

@slope.jit
def train_step(x, w, y, lr):
    y_hat = model_forward(x, w, b)
    loss, (gw,) = gloss_fn(y_hat, y)
    w = w - lr * gw
    return w

N = 50
x = slope.randn(N, 2)
y = slope.randn(N, 1)
w = slope.randn(1, 5)
w0 = w
lr = slope.tensor([0.001])
for i in range(10):
    w = train_step(x, w, y, lr)

```

# Contributing
m
Fork this repo and hack, and maybe do a PR, too many things need to be done (see Roadmap)
Idk everything is flaky and I am still experimenting and doing many API changes, maybe later I will open a new github repo.

# Roadmap
- cuda
- onnxruntime backend
- numpy backend
- scatter gather ops
- gpt2 training
- whisper inference
- passing state random sampling threefry
- core tests, operators tests on all Trace types