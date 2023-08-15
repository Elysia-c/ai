import tensorflow as tf
import numpy as np

x=tf.Variable(3.)
y=tf.Variable(4.)

#自动求导
with tf.GradientTape() as tape:
    # y=tf.square(x)
    # z=pow(x,3)
    f=tf.square(x)+2*tf.square(y)+1

# dy_dx=tape.gradient(y,x)
# dz_dx=tape.gradient(z,x)
df_dy,df_dx=tape.gradient(f,[x,y])

print(f)
print(df_dx)
print(df_dy)
del tape