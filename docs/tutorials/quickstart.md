# Quickstart

There are many examples in [examples/](examples/) folder.

We start by running MNIST classifier training, [examples/nn/mnist_mlp.py](examples/nn/mnist_mlp.py)

```sh
python examples/nn/mnist_mlp.py
```

```sh
Starting training...
...
Train epoch: 2, batch: 299/300, loss: 12.51: 100%|██████████████████████████████████████████████████████████████████████| 300/300 [00:02<00:00, 139.36it/s]
Epoch 2 in 2.15 sec
Test set accuracy 0.97
```


By setting the `SLOPE_BACKEND` flag, we change the backend to either `iree` (default), `onnxruntime` and `numpy`.
We can also set `LOG_JIT=1` to verbose print the backend output.

```sh
LOG_JIT=1 SLOPE_BACKEND=onnxruntime python examples/nn/mnist_mlp.py
```

```sh
...

---- train_step codegen:

<ir_version: 7, opset_import: ["" : 18, "slope":1]>
main (float[100, 784] x0, float[10, 100] x1, float[200, 28, 28] x2, float[200, 10] x3, int32[] x4, float[100, 784] x5, float[10, 100] x6, float[] x7) => (float[] y0, float[100, 784] y2, float[10, 100] y4, int32[] y5, float[100, 784] y1, float[10, 100] y3, float[] x7)
{
    
    z0_shape = Constant <value = int64[2] { 200, 784 } >()
    z0 = Reshape(x2, z0_shape)
...
    y4 = Sub(x1, z164)
    z165_fill_value = Constant < value = int32[1] { 1 }>()
    z165_squeeze_dim = Constant <value = int64[1] {0}> ()
    z165 = Squeeze (z165_fill_value, z165_squeeze_dim)
    
    y5 = Add(x4, z165)
}
...

===============

Train epoch: 0, batch: 58/300, loss: 71.23:  20%|██████████████▎                                                          | 59/300 [00:01<00:04, 55.45it/s]
```


### Environment flags
put this before the command to set

```
# prints the jitted code
LOG_JIT=1 

# set device
DEFAULT_DEVICE=cpu
DEFAULT_DEVICE=cuda
DEFAULT_DEVICE=metal

# set backend
SLOPE_BACKEND=iree # iree backend (default)
SLOPE_BACKEND=onnxruntime # onnxruntime backend
SLOPE_BACKEND=numpy # numpy backend (extremely SLOW)
```
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

Operation calls are jitted as individual programs eagerly.
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