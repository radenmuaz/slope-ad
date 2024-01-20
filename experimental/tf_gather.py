import tensorflow as tf
import tensorflow.mlir.experimental as tme

# print('#1')
x = tf.constant([[0.,1.],[2.,3.]], dtype=tf.float32)
w = tf.constant([[1,0],[0,1]], dtype=tf.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = tf.gather_nd(x, w,0)
# print(f"{y=}")

f =  tme.convert_function((tf.function(tf.gather_nd)).get_concrete_function(x, w))
print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))
# print('\n#2')
# x = tf.constant([[0.,1.],[2.,3.]], dtype=slope.float32)
# w = tf.constant([[1],[0]]).cast(slope.int64)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather(w)
# print(f"{y=}")

# print('\n#3')
# x = tf.constant([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=slope.float32)
# w = tf.constant([[0,1],[1,0]], dtype=slope.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather(w)
# print(f"{y=}")


# print('\n#4')
# x = tf.constant([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=slope.float32)
# w = tf.constant([[[0,1]],[[1,0]]], dtype=slope.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather(w)
# print(f"{y=}")