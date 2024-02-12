'''
# run with different backends
SLOPE_BACKEND=iree python examples/nn/cifar10_convnet.py
SLOPE_BACKEND=onnxruntime python examples/nn/cifar10_convnet.py

# print backend code
LOG_JIT=1 SLOPE_BACKEND=iree python examples/nn/cifar10_convnet.py
'''

import slope
import slope.nn as nn

import time
import numpy as np
from tqdm import tqdm

import numpy as np
from lib.datasets.cifar10 import get_cifar10
from lib.models.cv.resnet_cifar import resnet

np.random.seed(12345)
@slope.jit
def train_step(model, batch, optimizer):
    def train_loss_fn(model, batch):
        x, y = batch
        logits, model = model(x, training=True)
        loss = logits.cross_entropy(y) / x.size(0)

        return loss, (model, logits)

    (loss, (model, logits)), gmodel = slope.value_and_grad(train_loss_fn, has_aux=True)(model, batch)
    model, optimizer = optimizer(model, gmodel)
    return loss, logits, model, optimizer, gmodel


@slope.jit
def test_step(model, batch):
    x, y = batch
    logits = model(x, training=False)
    loss = logits.cross_entropy(y) / x.size(0)
    return loss, logits


def get_dataloader(images, labels, batch_size, shuffle=False):
    N = images.shape[0]
    perm = np.random.permutation(N) if shuffle else np.arange(N)
    mean = slope.tensor([0.4914, 0.4822, 0.4465], dtype=slope.float32)[None, ..., None, None]
    std = slope.tensor([0.2023, 0.1994, 0.2010], dtype=slope.float32)[None, ..., None, None]

    @slope.jit
    def standardize(x):
        return (x - mean) / std

    def data_iter():
        for i in range(N // batch_size):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            x = standardize(slope.tensor(images[batch_idx], dtype=slope.float32))
            y = slope.tensor(labels[batch_idx], dtype=slope.int32)
            yield x, y

    nbatches = (images.shape[0] // batch_size) + (1 if (images.shape[0] % batch_size != 0) else 0)
    return data_iter, nbatches


if __name__ == "__main__":
    nepochs = 160
    train_batch_size = 100  # TODO: must be multiple of dataset
    test_batch_size = 100
    train_images, train_labels, test_images, test_labels = get_cifar10()

    model = resnet(depth=20)
    # optimizer = nn.AdamW(model, lr=1e-1,weight_decay=1e-5)
    optimizer = nn.SGD(model, lr=0.1, momentum=0.9, weight_decay=1e-4)

    train_dataloader, ntrain_batches = get_dataloader(train_images, train_labels, train_batch_size, shuffle=True)
    test_dataloader, ntest_batches = get_dataloader(test_images, test_labels, test_batch_size, shuffle=False)
    # scheduler = nn.MultiStepLR(milestones=[80*ntrain_batches, 120*ntrain_batches], gamma=0.1)

    print("\nStarting training...")
    best_acc = 0.0
    for epoch in range(nepochs):
        if epoch in [80, 120]:
            optimizer.hp.lr = optimizer.hp.lr * 0.1
            print(f"Optimzer lr now={optimizer.hp.lr.numpy()}")
        total_loss = 0.0
        true_positives = 0.0
        for i, batch in (pbar := tqdm(enumerate(train_dataloader()), total=ntrain_batches, leave=False)):
        #   with slope.core.Timing("RUN "):
            with slope.core.Timing("RUN "): loss, logits, model, optimizer, gmodel = train_step(model, batch, optimizer)
            # loss, logits, model, optimizer, gmodel = train_step(model, batch, optimizer)
            loss_numpy = float(loss.numpy())
            total_loss += loss_numpy
            y_hat, y = logits.argmax(-1), batch[1]
            true_positives += float((y_hat == y).cast(slope.float32).sum().numpy())
            train_loss = total_loss / (i + 1)
            train_acc = true_positives / (train_batch_size * (i + 1))
            msg = f"Train epoch: {epoch}, " f"batch_loss: {loss_numpy:.4f}, " f"loss: {train_loss:.4f}, acc: {train_acc:.4f}"
            pbar.set_description(msg)
        mins, secs = divmod(pbar.format_dict["elapsed"], 60)
        print(f"{msg}, Time taken: {int(mins)}:{secs:.2f}")

        total_loss = 0.0
        true_positives = 0.0
        for i, batch in (pbar := tqdm(enumerate(test_dataloader()), total=ntest_batches, leave=False)):
            (
                loss,
                logits,
            ) = test_step(model, batch)
            loss_numpy = float(loss.numpy())
            total_loss += loss_numpy
            y_hat, y = logits.argmax(-1), batch[1]
            true_positives += float((y_hat == y).cast(slope.float32).sum().numpy())
            test_loss = total_loss / (i + 1)
            test_acc = true_positives / (test_batch_size * (i + 1))
            msg = f"Test epoch: {epoch}, " f"batch_loss: {loss_numpy:.4f}, " f"loss: {test_loss:.4f}, acc: {test_acc:.4f}"
            pbar.set_description(msg)
        mins, secs = divmod(pbar.format_dict["elapsed"], 60)
        print(f"{msg}, Time taken: {int(mins)}:{secs:.2f}")
        if test_acc > best_acc:
            print(f"  Best accuracy {test_acc:.4f} at epoch {epoch}\n")
            best_acc = test_acc
