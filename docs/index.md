# SlopeAD

Slope is a small automatic differentation (AD) engine, focused on machine learning (ML)

This project is designed to be a hackable, educational AD engine focused on ML, yet able to do things end-to-end training to deployment, instead of just some simple toy examples.

Tensor semantics are similar to Pytorch, functional API is similar to [JAX](https://github.com/google/jax), tensor operators code is heavily derived from [tinygrad](https://tinygrad.org/).

Example:
```python
import slope

def f(x):
    y = x * 2.0
    return y.sum()

x = slope.tensor([1.,2.,3.])
gf_x = slope.grad(f)(x)
print(f"{gf_x=}")
```
```
gf_x=<Tensor: val=
[2. 2. 2.]
shape=(3,), dtype=float32, device='cpu:0'>
```


# Install

## Stable release

```
pip install slope-ad
```

## Latest

```
git clone https://github.com/radenmuaz/slope-ad
cd slope
pip install -e .
```

Or you can just copy `src/slope` to your projects.

# Docs

Head over to [quickstart](./tutorials/quickstart.md)

# Features

1. Forward-mode, reverse-mode, and higher-order AD.

2. Just-in-time compilation, with interchangeable backends, supporting CPU, CUDA and Metal:
    - [IREE](https://iree.dev/) (StableHLO MLIR)
    - [ONNX Runtime](https://onnxruntime.ai/) (ONNX)
    - NumPy (CPU-only)

3. Training and inference, examples:
    - [MLP on MNIST](examples/nn/mnist_mlp.py)
    - [ResNet on CIFAR-10](examples/nn/cifar_resnet.py)
    - [Export jitted function](examples/simple/export.py)

4. Small (?)
    - <3000 lines of core code [slope/core.py](./src/slope/core.py), after run with `black src --line-length 140`

5. Operators and procedures system
    - 33 core operators defined in [slope/operators.py](./src/slope/operators.py)
        - Unary: `exp log sin sqrt invert cast stop_gradient`
        - Binary: `add mul sub div pow equal less greater maximum`
        - Reduce: `sum max`
        - Shape: `reshape expand permute slice pad flip cat`
        - Init: `full arange random_normal random_uniform`
        - GeneralReduce: `matmul conv gather_nd scatter_nd`
    - Composite operators system with "procedures" [slope/procedures.py](./src/slope/procedures.py)
        - Procedures are functions containing calls to operators, exposed with `Tensor.procedure_name(*args)` syntax.
        - Useful for definitions like:
          - `x.cos()`, where `def cos(x): return (math.pi/2 - x).sin()`
          - `x.conv_transpose(w)`: where `def conv_transpose(x, w): ...` is a very long function.
    - An operator can be directly implemented as code translation to backend, or _fallback_ to a procedure, e.g. there is `conv` procedure in case the backend has no implementation for it.

6. Extensible
    - Add new backend by defining implementation translations [slope/backends](./src/slope/backends)
    - NN module [slope/nn.py](./src/slope/nn.py)



# Contributing

Open a PR, things on the roadmap below need to be done.

# Roadmap

- Docs
- Symbolic shape inference 
- Dynamic shape jit
- Optimizer filter frozen params
- vmap vjp and jvp to compute jacobian and hessian
- iree backend currently has fixed seed random, implement threefry and JAX-like random
- make things fast
- llama (gpt) training
- whisper inference
- core tests, operators tests on all Trace types