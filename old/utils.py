def swap(f):
    return lambda x, y: f(y, x)


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


map_ = map


def map(f, *xs):
    return list(map_(f, *xs))


zip_ = zip


def zip(*args):
    fst, *rest = args = map(list, args)
    n = len(fst)
    for arg in rest:
        assert len(arg) == n
    return list(zip_(*args))
