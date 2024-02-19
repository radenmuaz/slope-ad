![logo](./assets/logo.jpeg)
# SlopeAD

Slope is a small automatic differentation (AD) engine, focused on machine learning (ML), supporting forward, reverse and higher-order AD.

This project is designed to be a **small**, **hackable** and **educational** AD engine focused on ML, yet able to do things **end-to-end from training to deployment**, instead of just some simple toy examples.

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


# Features

1. Functional API for forward-mode, reverse-mode, and higher-order AD, like in JAX:
    - `grad vjp jvp jit vmap`
    - `register_node tree_flatten tree_unflatten`


2. Just-in-time compilation, where code is compiled to these supported backends running on either CPU, CUDA and Metal:
    - [ONNX Runtime](https://onnxruntime.ai/) (ONNX graph)
    - [OpenXLA IREE](https://iree.dev/) (StableHLO MLIR)
    - NumPy (Python code)

3. Training and inference, examples:
    - [MLP on MNIST](examples/nn/mnist_mlp.py)
    - [ResNet on CIFAR-10](examples/nn/cifar_resnet.py)
    - [Export jitted function](examples/simple/export.py)

4. Small (?)
    - <3000 lines of core code [slope/core.py](./src/slope/core.py), after `black src --line-length 140`

5. Operators and procedures system
    - 33 core operators defined in [slope/operators.py](./src/slope/operators.py)
        - Unary: `exp log sin sqrt invert cast stop_gradient`
        - Binary: `add mul sub div pow equal less greater maximum`
        - Reduce: `sum max`
        - Shape: `reshape expand permute slice pad flip cat`
        - Init: `full arange random_normal random_uniform`
        - GeneralReduce: `matmul conv gather_nd scatter_nd`
    - Composite operators system with "procedures" [slope/procedures.py](./src/slope/procedures.py)
        - For defining Tensor functions composed with core operators, e.g.
          - `x.cos()`, where `def cos(x): return (math.pi/2 - x).sin()`
          - `x.conv_transpose(w)`: where `def conv_transpose(x, w, ... ): ...` is a very long function.
        - Procedures are exposed with `Tensor.procedure_name(*args)` syntax.
        

6. Extensible
    - Add new backend by defining implementation translations [slope/backends](./src/slope/backends)
    - Define new modules with NN module [slope/nn.py](./src/slope/nn.py)



# Docs

## Tutorials

[Quickstart](./docs/tutorials/quickstart.md): How Tensors work, how to write and jit compile functions and train something.

[NN Training](./docs/tutorials/nn_training.md): NN module overview.

[Internals Walkthrough](./docs/tutorials/internals_walkthrough.md): Understand the core of SlopeAD (hint: like JAX). Useful if you want to start contributing to SlopeAD

[Extending SlopeAD](./docs/tutorials/internals_walkthrough.md): Add new backend, operators, procedures. Modify the core functions.

## Examples

Examples are shorter and have less explanation, more code

[Linear Regression](./docs/tutorials/linear_regression.md)

[MLP on MNIST](./docs/tutorials/linear_regression.md)

## API reference

Docs are available online at [radenmuaz.github.io/slope-ad](https://radenmuaz.github.io/slope-ad)

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