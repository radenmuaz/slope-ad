```
   /|s
  / |l
 /  |o
/___|p 
 slope
```

Slope is a small automatic differentation (AD) engine, focused on machine learning (ML).
It is written to be educational and hackable, yet does things end-to-end, from training to deployment.
- Tensor API like Pytorch, 
- Higher-order derivaives, functional API and pytrees like JAX
- Operator decompositions and easily add new backend like tinygrad.


# Install

Clone this repo then
```
pip install -e .
```

Or you can just copy `src/slope` to your projects

# Quickstart

Slope is an automatic diffentation (AD) library, where the Tensor semantics feel like pytorch, but AD semantics feels like JAX (maybe also functorch)

# Slope have familiar Pytorch-like syntax

Most of the things you aer familiar in Pytorch works in Slope, probably.

```python
import slope
x = slope.ones(2, 5)
w = slope.arange(15).reshape(5,3)
b = slope.tensor([1., 2., 3.], dtype=slope.float32)
y = x @ w
print(y)
```

```

```

# Every operations are compiled with slope.jit


Actually these lines when run, are jitted as individual programs eagerly

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
# Also works if prefer not using @slope.jit decorator
#f = slope.jit(f)

# Then call the jitted function
x = slope.full((1,), 2.)
y = f(x)
```

```

```

# Reveal contents of jitted function

A decorated function with `slope.jit` is an instance object of `slope.core.Machine.jit`,
It has several utility methods for printing the generated backend code and exporting the function

```python
f_jitobj = f.get_jit_object(x)
print(f_jitobj.code)
f_jitobj.export('./f_export_folder') # see the folder of what are exported.
```

# Derivatives and gradients

Slope has several AD functions, like `slope.jvp` `slope.jvp` and `slope.grad`

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

# That's about it

Check out examples in `examples/simple` for more short snippets.

`slope/nn.py` provides example nn.Module system like in Pytorch, but with usage API like JAX
MNIST classifier example is in `example/mnist_mlp.py`

You are encourage to read the source, starting from `slope/__init__.py`, `slope/core.py` and `slope/backends/numpy_backend.py`
