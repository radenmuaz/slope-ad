import slope

x = slope.ones(())

def get_treedef(tree):
    return slope.M().tree_flatten(tree)[1]
# print(get_treedef((x,)))
print(get_treedef((x, x, (x, x))))