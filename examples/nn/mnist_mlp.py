import slope
import slope.nn as nn

import time
import itertools
import math
import numpy as np
from tqdm import tqdm

from lib.datasets.mnist import get_mnist

import numpy as np


def loss_fn(model, batch):
    x, y_onehot = batch
    preds = model(x)
    return -(preds * y_onehot).sum()


value_and_grad_loss_fn = slope.value_and_grad(loss_fn)


# @slope.jit
def train_step(model, batch, optimizer):
    loss, grad_loss_model = value_and_grad_loss_fn(model, batch)
    new_model, new_optimizer = optimizer(model, grad_loss_model)
    return loss, new_model, new_optimizer


def test_all(model, x, y):
    out = model(x)
    y_hat = out.argmax(-1)
    corrects = (y_hat == y).cast(slope.float32)
    accuracy = corrects.mean().numpy()
    return accuracy


if __name__ == "__main__":
    num_epochs = 3
    batch_size = 200  # TODO: must be multiple of dataset
    train_images, train_labels, test_images, test_labels = get_mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    log_interval = 10
    model = nn.Sequential(
            nn.Fn(lambda x: x.reshape(shape=(x.shape[0], math.prod(x.shape[1:])))),
            nn.MLP(784, 100, 10),
            nn.Fn(lambda x: x.log_softmax(axes=-1)),
    )
    optimizer = nn.SGD(model, lr=1e-3, momentum=0.8, weight_decay=1e-5)
    # optimizer = nn.Adam(model, lr=1e-4)

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                x = slope.tensor(train_images[batch_idx])
                y_onehot = slope.tensor(train_labels[batch_idx]).one_hot(10).cast(slope.float32)
                yield x, y_onehot

    x_test, y_test = slope.tensor(test_images), slope.tensor(test_labels).cast(slope.int32)

    batches = data_stream()
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for i in (pbar := tqdm(range(num_batches))):
            batch = next(batches)
            loss, model, optimizer = train_step(model, batch, optimizer)
            pbar.set_description(f"Train epoch: {epoch}, batch: {i}/{num_batches}, loss: {loss.numpy():.2f}")
        epoch_time = time.time() - start_time

        test_acc = test_all(model, x_test, y_test)
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Test set accuracy {test_acc:0.2f}")
