import slope


def mnist_slope_init():
    from slope.environments.v1 import v1_environment

    return slope.core.Machine(environment=v1_environment)


slope.set_slope_init(mnist_slope_init)
from slope import environment as sev
from slope.nn import init, layers, optim

import time
import itertools
import math
import numpy as np
from tqdm import tqdm

import tensor
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np


_DATA = "/tmp/slope_data/"


def download(url, filename):
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print(f"downloaded {url} to {_DATA}")


def one_hot(x, k, dtype=np.float32):
    return np.tensor(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.tensor(tensor.tensor("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.tensor(tensor.tensor("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(permute_train=False):
    train_images, train_labels, test_images, test_labels = mnist_raw()

    train_images = train_images / np.float32(255.0)
    test_images = test_images / np.float32(255.0)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)

    if permute_train:
        perm = np.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels


class Linear(slope.core.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        self.weight = sev.randn((out_dim, in_dim))
        self.bias = sev.zeros(out_dim) if bias else None

    def __call__(self, x):
        x = x.dot(self.weight.T())
        return x + self.bias if self.bias is not None else x


class MLP(slope.core.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        self.linear1 = Linear(in_dim, hid_dim)
        self.linear2 = Linear(hid_dim, out_dim)

    def __call__(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x

@slope.jit
def train_step(model, batch, optimizer):
    def loss_fn(model, batch):
        inputs, targets = batch
        preds = model(inputs)
        return -(preds * targets).sum()
    
    g_loss_fn = slope.grad(loss_fn)
    loss, g_model = g_loss_fn(model, batch)
    model, optimizer = optimizer(model, g_model)
    return loss, model, optimizer


def accuracy(model, batch):
    inputs, targets = batch
    target_class = np.argmax(targets.numpy(), axis=-1)
    predicted_class = np.argmax(model(inputs).numpy(), axis=-1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    step_size = 0.001
    num_epochs = 30
    batch_size = 200  # TODO: must be multiple of dataset.
    momentum_mass = 0.9

    model = MLP(784, 100, 10)

    train_images, train_labels, test_images, test_labels = mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    log_interval = num_batches // 4

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield sev.tensor(train_images[batch_idx]), sev.tensor(train_labels[batch_idx])

    batches = data_stream()
    g_loss_fn = slope.grad(loss_fn, ret_fval=True)

    def update(i, opt_state, batch):
        params = get_params(opt_state)
        loss, (g_params, _) = g_loss_fn(params, batch)
        return loss, opt_update(i, g_params, opt_state)

    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in tqdm(range(num_batches)):
            loss, opt_state = update(next(itercount), opt_state, next(batches))
            if i % log_interval == 0:
                print(f"loss: {loss.val:.2f}")
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        test_acc = accuracy(
            params,
            (
                sev.tensor(test_images),
                sev.tensor(test_labels),
            ),
        )
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Test set accuracy {test_acc}")
