# import ast

# def pretty_print_ast(node, indent=0):
#     indent_str = " " * indent
#     node_str = indent_str + ast.dump(node)
#     print(node_str)
#     for child_node in ast.iter_child_nodes(node):
#         pretty_print_ast(child_node, indent=indent + 4)

# # Example AST node
# tree = ast.parse("x = 1 + 2")

# # Pretty print the AST node
# pretty_print_ast(tree)

# @dataclass
# class NumpyConst(BaseConst):
#     val: Any


# @dataclass
# class NumpyParam(BaseParam):
#     id: int
#     val: Any
