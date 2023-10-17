import slope


def mnist_slope_init():
    from slope.environments.v1 import v1_environment

    return slope.core.Machine(environment=v1_environment)


slope.set_slope_init(mnist_slope_init)

import slope.nn as nn

import time
import itertools
import math
import numpy as np
from tqdm import tqdm

import array
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
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist_raw():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

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


def loss_fn(model, batch):
    inputs, targets = batch
    preds = model(inputs)
    return -(preds * targets).sum()


g_loss_fn = slope.grad(loss_fn, ret_fval=True)


@slope.jit
def train_step(model, batch, optimizer):
    loss, g_model = g_loss_fn(model, batch)
    new_model, new_optimizer = optimizer(model, g_model)
    return loss, new_model, new_optimizer


def accuracy(model, batch):
    inputs, targets = batch
    target_class = np.argmax(targets.numpy(), axis=-1)
    predicted_class = np.argmax(model(inputs).numpy(), axis=-1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    num_epochs = 3
    batch_size = 200  # TODO: must be multiple of dataset.

    train_images, train_labels, test_images, test_labels = mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    log_interval = num_batches // 4
    model = nn.Serial(
        [
            nn.Fn(lambda x: x.reshape(shape=(x.shape[0], math.prod(x.shape[1:])))),
            # nn.Linear(784, 10),
            nn.MLP(784, 100, 10),
            nn.Fn(lambda x: x.log_softmax(axes=-1)),
        ]
    )
    optimizer = nn.SGD(model, lr=1e-3, momentum=0.8, weight_decay=1e-5)

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield slope.tensor(train_images[batch_idx]), slope.tensor(train_labels[batch_idx])

    batches = data_stream()
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in tqdm(range(num_batches)):
            batch = next(batches)
            loss, model, optimizer = train_step(model, batch, optimizer)
            if i % log_interval == 0:
                print(f"loss: {loss.val:.2f}")
        epoch_time = time.time() - start_time

        test_acc = accuracy(
            model,
            (
                slope.tensor(test_images),
                slope.tensor(test_labels),
            ),
        )
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Test set accuracy {test_acc}")
