import tensorflow as tf

model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8,activation="relu",input_shape=(4,)))
model.add(tf.keras.layers.Dense(4,activation="relu"))
model.add(tf.keras.layers.Dense(3,activation="softmax"))
model.summary()
