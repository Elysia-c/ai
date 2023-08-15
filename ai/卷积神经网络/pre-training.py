from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

covn_base=ResNet50(include_top=False,
                   weights='imagment',
                   input_shape=(224,224,3))
covn_base.trainable=False

model=tf.keras.Sequential()
model.add(covn_base)
model.add(tf.keras.layers.GlobalveragePooling2D())
model.add(tf.keras.layers.Dense(10,activation="softmax"))