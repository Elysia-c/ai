import tensorflow as tf
import matplotlib.pyplot as plt

mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()

plt.axis("off")
plt.imshow(train_x[0],cmap="gray")
plt.show()