import os, gzip
import numpy as np
from . import utils
import struct
import array


def get_mnist_raw():
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
        utils.download(base_url + filename, filename)

    train_images = parse_images(os.path.join(utils._DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(os.path.join(utils._DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(os.path.join(utils._DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(os.path.join(utils._DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def get_mnist():
    train_images, train_labels, test_images, test_labels = get_mnist_raw()
    train_images = train_images / np.float32(255.0)
    train_labels = np.array(train_labels)
    test_images = test_images / np.float32(255.0)
    test_labels = np.array(test_labels)
    return train_images, train_labels, test_images, test_labels
