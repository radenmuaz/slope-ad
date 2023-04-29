import slope
from slope import ops
from nn import datasets, layers, optimizers, initializers



import time
import itertools

import numpy.random as npr

# from jax import jit, grad, random
# from jax.example_libraries import optimizers
# from jax.example_libraries import stax
# from jax.example_libraries.stax import Dense, Relu, LogSoftmax


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -ops.mean(ops.sum(preds * targets, axis=1))

def accuracy(params, batch):
  inputs, targets = batch
  target_class = ops.argmax(targets, axis=1)
  predicted_class = ops.argmax(predict(params, inputs), axis=1)
  return ops.mean(predicted_class == target_class)

init_random_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)

if __name__ == "__main__":
    rng = random.PRNGKey(0)
    train_images, train_labels, test_images, test_labels = datasets.mnist()

    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]
    batches = data_stream()

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    _, init_params = init_random_params(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
        opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc}")
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