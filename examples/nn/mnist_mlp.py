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
    return -(preds.log_softmax(-1) * y_onehot).sum()


value_and_gloss_fn = slope.value_and_grad(loss_fn)


@slope.jit
def train_step(model, batch, optimizer):
    loss, gloss_model = value_and_gloss_fn(model, batch)
    new_model, new_optimizer = optimizer(model, gloss_model)
    return loss, new_model, new_optimizer


@slope.jit
def test_all(model, x, y):
    out = model(x)
    y_hat = out.argmax(-1)
    corrects = (y_hat == y).cast(slope.float32)
    accuracy = corrects.mean()
    return accuracy


class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, act=nn.ReLU()):
        self.flatten_fn = nn.Fn(lambda x: x.reshape(shape=(x.shape[0], math.prod(x.shape[1:]))))
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, out_dim)
        self.act = act

    def __call__(self, x):
        x = self.flatten_fn(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    num_epochs = 3
    batch_size = 200  # TODO: must be multiple of dataset
    train_images, train_labels, test_images, test_labels = get_mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    log_interval = 10
    model = Net(in_dim=784, hid_dim=100, out_dim=10, act=nn.ReLU())
    optimizer = nn.SGD(model, lr=1e-3, momentum=0.9, weight_decay=1e-4)
    # optimizer = nn.Adam(model, lr=1e-3)

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
            # if i % 10 == 0: print(f"Train epoch: {epoch}, batch: {i}/{num_batches}, loss: {loss.numpy():.2f}")
        epoch_time = time.time() - start_time

        test_acc = test_all(model, x_test, y_test)
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Test set accuracy {test_acc.numpy():0.2f}")
