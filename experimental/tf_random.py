import tensorflow as tf
import tensorflow.mlir.experimental as tme

print('\n#1')
shape=tf.constant(1, shape=(1,))
y = tf.random.normal(shape); print(f"{y=}")
f =  tme.convert_function((tf.function(tf.random.normal)).get_concrete_function(shape))
print(tme.run_pass_pipeline(f, "tf-lower-to-mlprogram-and-hlo"))
