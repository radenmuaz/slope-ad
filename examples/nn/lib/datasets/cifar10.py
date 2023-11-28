import slope
from slope.core import Tensor
from pathlib import Path
import tarfile
import pickle
import os
import numpy as np
from urllib import request
from tqdm import tqdm

_DATA = "/tmp/slope_data/cifar10/"


def get_cifar10():
    base_url = "https://www.cs.toronto.edu/~kriz/"
    filename = "cifar-10-python.tar.gz"
    url = base_url + filename
    save_path = _DATA + filename
    if not os.path.exists(_DATA):
        os.makedirs(_DATA, exist_ok=True)
        if not os.path.isfile(save_path):
            print(f"downloading {url}")
            request.urlretrieve(url, save_path)
            print(f"saved {save_path}")

    def get_split(db_list):
        xs, ys = [], []
        for db in tqdm(db_list):
            x, y = db[b"data"], np.array(db[b"labels"])
            assert x.shape[0] == y.shape[0]
            xs += [x]
            ys += [y]
        images = np.concatenate(xs, 0).reshape((-1, 3, 32, 32)).astype(np.float32) / 255.0
        labels = np.concatenate(xs, 0).reshape((-1, 3, 32, 32))
        return images, labels

    tt = tarfile.open(save_path, mode="r:gz")
    db_list_train = [
        pickle.load(tt.extractfile(f"cifar-10-batches-py/data_batch_{i}"), encoding="bytes") for i in range(1, 6)
    ]
    db_list_test = [pickle.load(tt.extractfile("cifar-10-batches-py/test_batch"), encoding="bytes")]
    train_images, train_labels = get_split(db_list_train)
    test_images, test_labels = get_split(db_list_test)
    return train_images, train_labels, test_images, test_labels


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = get_cifar10()
    breakpoint()
