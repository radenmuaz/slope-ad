# SlopeAD

Slope is a small automatic differentation (AD) engine, focused on machine learning (ML)

This project is designed to be a **hackable**, **educational** AD engine focused on ML, yet able to do things **end-to-end from training to deployment**, instead of just some simple toy examples.

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


# Features

1. Functional API for forward-mode, reverse-mode, and higher-order AD, like in JAX:
    - `grad vjp jvp jit vmap`
    - `register_node tree_flatten tree_unflatten`

3. Just-in-time compilation, code is compiled to one of these backends: [ONNX Runtime](https://onnxruntime.ai/), [OpenXLA IREE](https://iree.dev/) and NumPy.

4. Training and inference, like [MLP on MNIST](examples/nn/mnist_mlp.py), [ResNet on CIFAR-10](examples/nn/cifar_resnet.py) and [export jitted function](examples/simple/export.py).

5. Small (?), less than 3000 lines of core code [slope/core.py](./src/slope/core.py), after `black src --line-length 140`

6. Operators and procedures system
    - 33 core operators defined in [slope/operators.py](./src/slope/operators.py)
    - `exp log sin sqrt invert cast stop_gradient add mul sub div pow equal less greater maximum sum max reshape expand permute slice pad flip cat full arange random_normal random_uniform matmul conv gather_nd scatter_nd`
    - Composite operators system with "procedures" [slope/procedures.py](./src/slope/procedures.py), for defining functions like `cos`, `conv` and `avgpool2d` as functions calling core operators.

7. Extensible, by writing new backend by defining implementation translations [slope/backends](./src/slope/backends), and adding more modules using [slope/nn.py](./src/slope/nn.py)


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

# Usage

## Tutorials

[Quickstart Tutorial](./docs/tutorials/quickstart.md): How Tensors work, how to write and jit compile functions and train something

[NN Training](./docs/tutorials/quickstart.md): Train a MLP on MNIST and ResNet on CIFAR-10

[Internals Walkthrough](./docs/tutorials/internals_walkthrough.md): Understand the core of SlopeAD (hint: like JAX)

[Extending SlopeAD](./docs/tutorials/internals_walkthrough.md): Add new backend, operators, procedures. Modify the core functions.

## Docs and API reference

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