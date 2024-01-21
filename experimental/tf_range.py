import tensorflow as tf
import tensorflow.mlir.experimental as tme

print('\n#1')
start=tf.constant(0)
limit=tf.constant(10)
delta=tf.constant(2, dtype=tf.int32)
y = tf.range(start, limit, delta); print(f"{y=}")
f =  tme.convert_function((tf.function(tf.range)).get_concrete_function(start, limit, delta))
print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))
