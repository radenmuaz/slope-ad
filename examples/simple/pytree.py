import slope


def get_treedef(tree):
    return slope.M().tree_flatten(tree)[1]


x = slope.ones(())
data = (x, x, (x, x))
treedef = get_treedef(data)
print(treedef)
breakpoint()
