import slope
from slope import ops
from slope.nn import init, layers, optim

import time
import itertools

import numpy as np

# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets used in examples."""


import array
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np


_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    """Download, parse and process MNIST data to unit scale and one-hot labels."""
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = _partial_flatten(train_images).astype(np.float32) / np.float32(255.0)
    test_images = _partial_flatten(test_images).astype(np.float32) / np.float32(255.0)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


def loss_fn(params, batch):
    inputs, targets = batch

    preds = predict(params, inputs)
    return -ops.reduce_sum(preds * targets, axis=(0, 1))


#   return -ops.mean(ops.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = np.argmax(targets, axis=-1)
    predicted_class = np.argmax(predict(params, inputs), axis=-1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    # init_random_params, predict = layers.serial(layers.Dense(10), layers.Softmax)
    init_random_params, predict = layers.serial(layers.Dense(10))
    init_random_params, predict = layers.serial(layers.Dense(10), layers.LogSoftmax)
    _, init_params = init_random_params((-1, 28 * 28))

    step_size = 0.001
    num_epochs = 10
    batch_size = 32
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                # yield train_images[batch_idx][0], train_labels[batch_idx][0]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    opt_init, opt_update, get_params = optim.sgd(step_size)
    opt_state = opt_init(init_params)

    def update(i, opt_state, batch):
        params = get_params(opt_state)
        loss, (g_params, _) = slope.ad.grad(loss_fn)(params, batch)
        # breakpoint()
        return loss, opt_update(i, g_params, opt_state)

    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in range(num_batches):
            loss, opt_state = update(next(itercount), opt_state, next(batches))
            # print(f'{i}, loss: {loss:.2f}')
            if i == 100:
                break
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        # train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        # print(f"Training set accuracy {train_acc}")
        print(f"Test set accuracy {test_acc}")

# dot product
# x = np.random.randn(1,3)
# y = np.random.randn(2,3)

# def f(x, y):
#     out = x
#     out = ops.dot(out, ops.T(y))
#     out = ops.softmax(out, axis=(1,))
#     out = ops.reduce_sum(out, axis=(0,1))
#     return out

# out, grad_out = slope.grad(f)(x, y)
# print(x)
# print(y)
# print(out)
# print(grad_out)


# def f(x,y):
#     out = ops.dot(x, ops.T(y))
#     # out = ops.mul(x,y)
#     return out

# x_dot=y_dot=np.array([[1,1,1],[1,1,1]])
# p, t= slope.jvp(f, (x,y), (x_dot,y_dot))
#
# print(p)
# print(t)
