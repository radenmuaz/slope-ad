import tensorflow as tf
import tensorflow.mlir.experimental as tme

# print('\n#1')
# x = tf.constant([[0.,1.],[2.,3.]], dtype=tf.float32);print(f"{x=}")
# w = tf.constant([[1,0],[0,1]], dtype=tf.int32);print(f"{w=}")
# y = tf.gather_nd(x, w,0); print(f"{y=}")
# f =  tme.convert_function((tf.function(tf.gather_nd)).get_concrete_function(x, w, 0))
# print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))


# print('\n#2')
# x = tf.constant([[0.,1.],[2.,3.]], dtype=tf.float32);print(f"{x=}")
# w = tf.constant([[1],[0]], dtype=tf.int32);print(f"{w=}")
# y = tf.gather_nd(x, w, 0); print(f"{y=}")
# f =  tme.convert_function((tf.function(tf.gather_nd)).get_concrete_function(x, w,0 ))
# print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))

# print('\n#3')
# x = tf.constant([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=tf.float32); print(f"{x=}")
# w = tf.constant([[0,1],[1,0]], dtype=tf.int32); print(f"{w=}")
# y = tf.gather_nd(x, w, 0); print(f"{y=}")
# f =  tme.convert_function((tf.function(tf.gather_nd)).get_concrete_function(x, w, 0))
# print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))


# print('\n#4')
# x = tf.constant([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=tf.float32); print(f"{x=}")
# w = tf.constant([[0,1],[1,0]], dtype=tf.int32); print(f"{w=}")
# y = tf.gather_nd(x, w, 0); print(f"{y=}")
# f =  tme.convert_function((tf.function(tf.gather_nd)).get_concrete_function(x, w, 0))
# print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))

print('\n#5')
x = tf.constant([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=tf.float32); print(f"{x=}")
w = tf.constant([[1],[0]], dtype=tf.int32); print(f"{w=}")
y = tf.gather_nd(x, w, 1); print(f"{y=}")
f =  tme.convert_function((tf.function(tf.gather_nd)).get_concrete_function(x, w, 1))
print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))

# def translate_gather_nd_to_stablehlo(x, indices, axis):
#     input_rank = len(x.shape)
#     indices_rank = len(indices.shape)
    
#     # Determine the number of dimensions to collapse
#     collapsed_dims = list(range(indices_rank - 1))
    
#     # Determine the offset_dims and start_index_map based on the axis
#     offset_dims = [axis]
#     start_index_map = [axis]
    
#     # Translate indices for stablehlo.gather
#     translated_indices = indices
#     if indices_rank < input_rank:
#         translated_indices = tf.concat([tf.zeros_like(indices[:, :1]), indices], axis=1)
    
#     # Create the slice_sizes tensor for stablehlo.gather
#     slice_sizes = tf.concat([tf.ones_like(offset_dims), tf.shape(translated_indices)[1:]], axis=0)

#     params = {
#         'input_tensor': x,
#         'indices_tensor': translated_indices,
#         'dimension_numbers': {
#             'offset_dims': offset_dims,
#             'collapsed_slice_dims': collapsed_dims,
#             'start_index_map': start_index_map,
#             'index_vector_dim': indices_rank - 1,
#         },
#         'indices_are_sorted': False,
#         'slice_sizes': slice_sizes,
#     }

#     return params
# translated_result = translate_gather_nd_to_stablehlo(x, w, 0)
# print(translated_result)

# # Example #2
# x2 = tf.constant([[0., 1.], [2., 3.]], dtype=tf.float32)
# w2 = tf.constant([[1], [0]], dtype=tf.int32)
# axis2 = 0
# mlir_code2 = translate_gather_nd_to_stablehlo(x2, w2, axis2)
# print(mlir_code2)

# # Example #3
# x3 = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=tf.float32)
# w3 = tf.constant([[0, 1], [1, 0]], dtype=tf.int32)
# axis3 = 0
# mlir_code3 = translate_gather_nd_to_stablehlo(x3, w3, axis3)
# print(mlir_code3)

# # Example #4
# x4 = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=tf.float32)
# w4 = tf.constant([[0, 1], [1, 0]], dtype=tf.int32)
# axis4 = 0
# mlir_code4 = translate_gather_nd_to_stablehlo(x4, w4, axis4)
# print(mlir_code4)

# # Example #5
# x5 = tf.constant([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=tf.float32)
# w5 = tf.constant([[1], [0]], dtype=tf.int32)
# axis5 = 1
# mlir_code5 = translate_gather_nd_to_stablehlo(x5, w5, axis5)
# print(mlir_code5)
