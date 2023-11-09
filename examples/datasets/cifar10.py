def fetch_cifar():
  X_train = Tensor.empty(50000, 3*32*32, device=f'disk:/tmp/cifar_train_x', dtype=dtypes.uint8)
  Y_train = Tensor.empty(50000, device=f'disk:/tmp/cifar_train_y', dtype=dtypes.int64)
  X_test = Tensor.empty(10000, 3*32*32, device=f'disk:/tmp/cifar_test_x', dtype=dtypes.uint8)
  Y_test = Tensor.empty(10000, device=f'disk:/tmp/cifar_test_y', dtype=dtypes.int64)

  if not os.path.isfile("/tmp/cifar_extracted"):
    def _load_disk_tensor(X, Y, db_list):
      idx = 0
      for db in db_list:
        x, y = db[b'data'], np.array(db[b'labels'])
        assert x.shape[0] == y.shape[0]
        X[idx:idx+x.shape[0]].assign(x)
        Y[idx:idx+x.shape[0]].assign(y)
        idx += x.shape[0]
      assert idx == X.shape[0] and X.shape[0] == Y.shape[0]

    print("downloading and extracting CIFAR...")
    fn = Path(__file__).parent.resolve() / "cifar-10-python.tar.gz"
    download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', fn)
    tt = tarfile.open(fn, mode='r:gz')
    _load_disk_tensor(X_train, Y_train, [pickle.load(tt.extractfile(f'cifar-10-batches-py/data_batch_{i}'), encoding="bytes") for i in range(1,6)])
    _load_disk_tensor(X_test, Y_test, [pickle.load(tt.extractfile('cifar-10-batches-py/test_batch'), encoding="bytes")])
    open("/tmp/cifar_extracted", "wb").close()

  return X_train, Y_train, X_test, Y_test
