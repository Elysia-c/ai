import tensorflow as tf
tf.random.set_seed(8)
with tf.device2('/CPU:0'):
    a=tf.config.experimental.list_physical_devices(device_type="GPU")
    b=tf.reshape(a,[2,3,4])
    c=tf.matmul(a,b)
print(c)