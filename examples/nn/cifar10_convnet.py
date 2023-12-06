import slope
import slope.nn as nn

import time
import numpy as np
from tqdm import tqdm

import numpy as np
from lib.datasets.cifar10 import get_cifar10
from lib.models.cv.resnet_cifar import resnet


@slope.jit
def train_step(model, batch, optimizer):
    def train_loss_fn(model, batch):
        x, y = batch
        logits, model = model(x, training=True)
        loss = -(logits.log_softmax() * y).sum()
        return loss, (model, logits)
    (loss, (model, logits)), grad_model = slope.value_and_grad(train_loss_fn, has_aux=True)(model, batch)
    model, new_optimizer = optimizer(model, grad_model)
    return loss, logits, model, new_optimizer

@slope.jit
def test_step(model, batch):
    x, y = batch
    logits = model(x, training=False)
    loss = -(logits.log_softmax() * y).sum()
    return loss, logits

def get_dataloader(images, labels, batch_size, transforms_fn, shuffle=False):
    N = images.shape[0]
    perm = np.random.permutation(N) if shuffle else np.arange(N)
    def data_iter():
        for i in range(N//batch_size):
            batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            x = transforms_fn(slope.tensor(images[batch_idx]))
            y_onehot = slope.tensor(labels[batch_idx]).one_hot(10).cast(slope.float32)
            yield x, y_onehot
    nbatches = images.shape[0] // batch_size + (1 if (images.shape[0] % batch_size != 0) else 0)
    return data_iter(), nbatches
        

def train_transforms_fn(x):
    mean = slope.tensor([0.4914, 0.4822, 0.4465])[None, ..., None, None]
    std = slope.tensor([0.2023, 0.1994, 0.2010])[None, ..., None, None]
    return (x - mean) / std


def test_transforms_fn(x):
    mean = slope.tensor([0.4914, 0.4822, 0.4465])[None, ..., None, None]
    std = slope.tensor([0.2023, 0.1994, 0.2010])[None, ..., None, None]
    return (x - mean) / std

if __name__ == "__main__":
    nepochs = 50
    train_batch_size = 50  # TODO: must be multiple of dataset
    test_batch_size = 50 
    train_images, train_labels, test_images, test_labels = get_cifar10()
    
    model = resnet(depth=8)
    # optimizer = nn.Adam(model, lr=1e-3)
    optimizer = nn.SGD(model, lr=0.1, momentum=0.9, weight_decay=1e-4)
    
    train_dataloader, ntrain_batches = get_dataloader(
        train_images, train_labels, train_batch_size, train_transforms_fn, shuffle=True)
    test_dataloader, ntest_batches = get_dataloader(
        test_images, test_labels, test_batch_size, test_transforms_fn, shuffle=False)

    print("\nStarting training...")
    best_acc = 0.
    acc = 0.
    for epoch in range(nepochs):
        # start_time = time.perf_counter_ns()
        total_loss = 0.
        true_positives = 0.
        for i, batch in (pbar := tqdm(enumerate(train_dataloader))):
            loss, logits, model, optimizer = train_step(model, batch, optimizer)
            total_loss += float(loss.numpy())
            y_hat, y =  logits.argmax(-1), batch[1].argmax(-1)
            true_positives += float((y_hat == y).cast(slope.float32).mean().numpy())
            N = train_batch_size * (i+1)
            msg = f"Train epoch: {epoch}, batch: {i}/{ntrain_batches}, loss: {total_loss/N:.4f}, acc: {true_positives/N:.4f}"
            pbar.set_description(msg)
            # if i == 3: break
        # epoch_time = (time.perf_counter_ns() - start_time)*1e-9

        total_loss = 0.
        true_positives = 0.
        for i, batch in (pbar := tqdm(enumerate(test_dataloader))):
            loss, logits, model, optimizer = train_step(model, batch, optimizer)
            total_loss += float(loss.numpy())
            y_hat, y =  logits.argmax(-1), batch[1].argmax(-1)
            true_positives += float((y_hat == y).cast(slope.float32).mean().numpy())
            msg = f"Test epoch: {epoch}, batch: {i}/{ntest_batches}, loss: {total_loss/N:.4f}, acc: {true_positives/N:.4f}"
            pbar.set_description(msg)
        corrects = (y_hat == y)
        acc = float(corrects.mean().numpy())
        if acc > best_acc:
            print(f"Best accuracy {acc:.2f} at epoch {epoch}")
            best_acc = acc

