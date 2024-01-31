import slope

# x = slope.zeros(8, dtype=slope.float32)
# print(f"before: {x=}")
# w = slope.tensor([[4], [3], [1], [7]], dtype=slope.int32) # iree
# # w = slope.tensor([[4], [3], [1], [7]], dtype=slope.uint64) # onnxruntime
# u = slope.tensor([9., 10., 11., 12.], dtype=slope.float32)
# y = slope.scatter_nd(x,w,u)
# print(f"{w=}")
# print(f"{u=}")
# print(f"{x=}")
# print(f"{y=}")




# u: Tensor of rank q + r - indices_shape[-1] - 1.
# updates tensor should have shape equal to indices.shape[:-1] + data.shape[indices.shape[-1]:]

# x = slope.zeros((2,2), dtype=slope.float32)
# print(f"before: {x=}")

# w = slope.tensor([[1,0],[0,1]], dtype=slope.int32)
# u = slope.tensor([1,2], dtype=slope.float32)

# y = slope.scatter_nd(x,w,u)
# print(f"{w=}")
# print(f"{u=}")
# print(f"{x=}")
# print(f"{y=}")

# w = slope.tensor([[0,0],[1,0],[0,1],[1,1]], dtype=slope.int64)
# u = slope.tensor([1,2,3,4], dtype=slope.float32)

# w = slope.tensor([[0],[1]], dtype=slope.int64)
# u = slope.tensor([[1,2],[3,4]], dtype=slope.float32)

# w = slope.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=slope.int64)
# u = slope.tensor([1,2,3,4], dtype=slope.float32)

'''
module {
  func.func @__inference_tensor_scatter_add_13(%arg0: tensor<8xf32>, %arg1: tensor<4x1xi32>, %arg2: tensor<4xf32>) -> tensor<8xf32> attributes {allow_soft_placement = false} {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {indices_are_sorted = false, 
    scatter_dimension_numbers = #stablehlo.scatter<
    inserted_window_dims = [0],
    scatter_dims_to_operand_dims = [0],
    index_vector_dim = 1
    >,
      unique_indices = false} : (tensor<8xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
'''

'''
func.func @main (%x0: tensor<8xf32>, %x1: tensor<4x1xi32>, %x2: tensor<4xf32>) -> (tensor<8xf32>)
{
    %y0 = "stablehlo.scatter"(%x0, %x1, %x2) ({
      ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %0 = "stablehlo.add"(%arg0, %arg1)  : (tensor<f32>,tensor<f32>) -> tensor<f32>
        "stablehlo.return"(%0) : (tensor<f32>) -> ()
    }) {
      scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
      indices_are_sorted = false,
      unique_indices = false
    }  : (tensor<8xf32>,tensor<4x1xi32>,tensor<4xf32>) -> tensor<8xf32>
    
    "func.return"(%y0): (tensor<8xf32>) -> ()
}
'''

x    = slope.tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=slope.float32)
w = slope.tensor([[0], [2]], dtype=slope.int32)
u = slope.tensor([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=slope.float32)
y = slope.scatter_nd(x,w,u)
print(y)


'''
module {
  func.func @__inference_tensor_scatter_nd_update_11(
    %arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi32>, 
    %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> attributes {allow_soft_placement = false} {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      stablehlo.return %arg4 : tensor<f32>
    }) {
        indices_are_sorted = false,
        scatter_dimension_numbers = #stablehlo.scatter<
        update_window_dims = [1, 2],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1>, unique_indices = false} : (tensor<4x4x4xf32>, tensor<2x1xi32>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
    return %0 : tensor<4x4x4xf32>
  }
}
'''
