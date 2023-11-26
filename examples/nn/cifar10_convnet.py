import slope


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

def loss_fn(model, batch):
    inputs, targets = batch
    preds = model(inputs)
    return -(preds * targets).sum()


g_loss_fn = slope.value_and_grad(loss_fn)


@slope.jit
def train_step(model, batch, optimizer):
    loss, (g_model, _) = g_loss_fn(model, batch)
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
    log_interval = 10
    # log_interval = num_batches // 4
    model = nn.Serial(
        [
            nn.Fn(lambda x: x.reshape(shape=(x.shape[0], math.prod(x.shape[1:])))),
            # nn.Linear(784, 10),
            nn.MLP(784, 100, 10),
            nn.Fn(lambda x: x.log_softmax(axes=-1)),
            # nn.Fn(lambda x: x.softmax(axes=-1)),
        ]
    )
    # optimizer = nn.SGD(model, lr=1e-9, momentum=0., weight_decay=0)
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
                print(f"loss: {loss.numpy():.2f}")
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
