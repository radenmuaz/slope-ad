import slope


import slope.nn as nn

import time
import itertools
import math
import numpy as np
from tqdm import tqdm

import numpy as np
from lib.datasets.cifar10 import get_cifar10


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        self.conv1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_dim)

        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential([
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(out_dim)
            ])
        else:
            self.shortcut = lambda x: x

    def __call__(self, x):
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = out.relu()
        return out

class Net(nn.Module):
    def __init__(self):
        self.block1 = ResnetBlock(3, 32)
        self.block2 = ResnetBlock(32, 32)
        self.avgpool = nn.AvgPool2d(16)
        self.linear = nn.Linear(128, 10)
        self.flatten_fn = nn.Fn(lambda x: x.reshape((x.shape[0], -1)))

    def __call__(self, x, training=False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = self.flatten_fn(x)
        x = self.linear(x)
        return x, self if self.training else x


def loss_fn(model, batch):
    inputs, targets = batch
    preds = model(inputs)
    return -(preds * targets).sum()


g_loss_fn = slope.value_and_grad(loss_fn, has_aux=True)


# @slope.jit
def train_step(model, batch, optimizer):
    (loss, model_), g_model = g_loss_fn(model, batch)
    breakpoint()
    new_model, new_optimizer = optimizer(model, g_model)
    return loss, new_model, new_optimizer


def accuracy(model, batch):
    inputs, targets = batch
    target_class = np.argmax(targets.numpy(), axis=-1)
    predicted_class = np.argmax(model(inputs).numpy(), axis=-1)
    return np.mean(predicted_class == target_class)


if __name__ == "__main__":
    num_epochs = 10
    batch_size = 50  # TODO: must be multiple of dataset
    train_images, train_labels, test_images, test_labels = get_cifar10()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    log_interval = 10
    # log_interval = num_batches // 4
    model = Net()
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
