import tensorflow as tf
import tensorflow.mlir.experimental as tme

import slope

7

# print('\n#1')
# x = tf.ones(8, dtype=tf.float32);print(f"{x=}")
# w = tf.constant([[4], [3], [1], [7]], dtype=tf.int32);print(f"{w=}")
# u = tf.constant([9., 10., 11., 12.], dtype=tf.float32);print(f"{u=}")
# y = tf.tensor_scatter_nd_add(x,w,u);print(f"{y=}")
# f =  tme.convert_function((tf.function(tf.tensor_scatter_nd_add)).get_concrete_function(x, w, u,0))
# print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))

'''
module {
  func.func @__inference_tensor_scatter_add_13(%arg0: tensor<8xf32>, %arg1: tensor<4x1xi32>, %arg2: tensor<4xf32>) -> tensor<8xf32> attributes {allow_soft_placement = false} {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<8xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
'''

# print('\n#2')
# x    = tf.constant([[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
#             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
#             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
#             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=tf.float32); print(f"{x=}")
# w = tf.constant([[0], [2]], dtype=tf.int32); print(f"{w=}")
# u = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
#             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=tf.float32); print(f"{u=}")

# y = tf.tensor_scatter_nd_update(x,w,u);print(f"{y=}")
# f =  tme.convert_function((tf.function(tf.tensor_scatter_nd_update)).get_concrete_function(x, w, u))
# print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))


# y = tf.tensor_scatter_nd_add(x,w,u);print(f"{y=}")
# f =  tme.convert_function((tf.function(tf.tensor_scatter_nd_add)).get_concrete_function(x, w, u))
# print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))


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
        scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1, 2],
        inserted_window_dims = [0],
        scatter_dims_to_operand_dims = [0],
        index_vector_dim = 1>, unique_indices = false} : (tensor<4x4x4xf32>, tensor<2x1xi32>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
    return %0 : tensor<4x4x4xf32>
  }
}
'''

## update
# output  = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
#             [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
#             [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
#             [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]

## add
# array([
# [[ 6.,  7.,  8.,  9.],
#         [11., 12., 13., 14.],
#         [15., 14., 13., 12.],
#         [12., 11., 10.,  9.]],

#        [[ 1.,  2.,  3.,  4.],
#         [ 5.,  6.,  7.,  8.],
#         [ 8.,  7.,  6.,  5.],
#         [ 4.,  3.,  2.,  1.]],

#        [[ 9.,  8.,  7.,  6.],
#         [ 6.,  5.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 9., 10., 11., 12.]],

#        [[ 8.,  7.,  6.,  5.],
#         [ 4.,  3.,  2.,  1.],
#         [ 1.,  2.,  3.,  4.],
#         [ 5.,  6.,  7.,  8.]]], dtype=float32)>

print('\n#3')
x = tf.zeros((2,2), dtype=tf.float32);print(f"{x=}")
w = tf.constant([[1,0],[0,1]], dtype=tf.int32);print(f"{w=}")
u = tf.constant([1,2], dtype=tf.float32);print(f"{u=}")
y = tf.tensor_scatter_nd_add(x,w,u);print(f"{y=}")
f =  tme.convert_function((tf.function(tf.tensor_scatter_nd_add)).get_concrete_function(x, w, u,0))
print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))
'''
func.func @__inference_tensor_scatter_add_13(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xi32>, %arg2: tensor<2xf32>) -> tensor<2x2xf32> attributes {allow_soft_placement = false} {
    %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = false} : (tensor<2x2xf32>, tensor<2x2xi32>, tensor<2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

'''