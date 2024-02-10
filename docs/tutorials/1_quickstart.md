# Slope internals tutorial

## Slope has familiar Pytorch-like syntax

Most of the things familiar in Pytorch works in Slope, probably.

```python
import slope
x = slope.ones(2, 5)
w = slope.arange(15, dtype=slope.float32).reshape(5,3)
b = slope.tensor([1., 2., 3.], dtype=slope.float32)
y = x @ w + b
print(y)
```

## Every operations are compiled with slope.jit


Actually when these lines are run, each operation calls jitted as individual programs eagerly.
Try running this on terminal:
```sh
LOG_JIT=1 python -c "import slope; print(slope.ones(3)*2)"
```

```sh
...
---- full_shape__lp__rp__fill_value_2_dt_0_dtype_<DType:float32>_device_<Device:'cpu:0'>_ codegen:

func.func @main () -> (tensor<f32>)
{
    %y0 = "stablehlo.constant"() { value = dense<2.0> : tensor<f32> } : () -> (tensor<f32>)
    "func.return"(%y0): (tensor<f32>) -> ()
}
... # plenty of outputs
....
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
print(y)
```

To see the actual code:
```python
jit_object = f.lower(x)
# slope Program intermediate representation
print(jit_object.program)
# backend code
print(jit_object.code)
```
```sh
def f(x0): # [1, f32] -> [1, f32]
    y0 = slope.mul(x0, x0) # ([1, f32], [1, f32]) -> [1, f32]
    return y0
```
```sh

func.func @main (%x0: tensor<1xf32>) -> (tensor<1xf32>)
{
    %y0 = "stablehlo.multiply"(%x0, %x0) : (tensor<1xf32>,tensor<1xf32>) -> (tensor<1xf32>)
    "func.return"(%y0): (tensor<1xf32>) -> ()
}
```

# Derivatives and gradients

Slope has several AD functions, like `slope.jvp` `slope.vjp` and `slope.grad`

To do the usual backprop things:
```python
def f(x, w):
    y = x @ w
    return y
def loss_fn(x, w, y):
    y_hat = f(x,w)
    return ((y_hat - y)**2).sum()
gloss_fn = slope.value_and_grad(loss_fn, argnums=(1,))

@slope.jit
def train_step(x, w, y, lr):
    loss, gw = gloss_fn(x, w, y)
    w = w - lr * gw
    return loss, w

N = 50
x = slope.randn(N, 2)
y = slope.randn(N, 1)
w = slope.randn(2, 1)
lr = slope.tensor([0.001])
for i in range(10):
    loss, w = train_step(x, w, y, lr)
    print(i, loss.numpy())

```

```
0 102.412125
1 88.60157
2 78.322815
3 70.644066
4 64.883766
5 60.54294
6 57.25553
7 54.75257
8 52.83598
9 51.359528
```

# Contributing

Fork this repo and hack, and maybe do a PR, too many things need to be done (see Roadmap)
Idk everything is flaky and I am still experimenting and doing many API changes, maybe later I will open a new github repo.

# Roadmap
- iree backend currently has fixed seed random, implement threefry and JAX-like random
- make things fast
- llama (gpt) training
- whisper inference
- core tests, operators tests on all Trace types