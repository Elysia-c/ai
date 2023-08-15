import tensorflow as tf
print("Tensorflow version",tf.__version__)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#加载数据
plt.rcParams["font.sans-serif"]="SimHei"
cifar10=tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train[0:2000]
y_train=y_train[0:2000]

y_train=tf.one_hot(y_train,10)
y_test=tf.one_hot(y_test,10)

y_train=np.squeeze(y_train)
y_test=tf.squeeze(y_test)
print(y_train[0])

#建立Sequential模型
model=tf.keras.Sequential()
#卷积层1 最大池化层1
model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding="same",activation="sigmoid",input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#卷积层2 最大池化层2
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding="valid",activation="sigmoid"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#卷积层3
model.add(tf.keras.layers.Conv2D(filters=120,kernel_size=5,padding="valid",activation="sigmoid"))
model.add(tf.keras.layers.Flatten())

#全连接层
model.add(tf.keras.layers.Dense(84,activation="sigmoid"))
#输出层
model.add(tf.keras.layers.Dense(10,activation="softmax"))


model.summary()

#配置训练方法
model.compile(optimizer='adam',# 优化器
              loss='categorical_crossentropy',# 损失函数
              metrics=['accuracy']# 准确值
              )

history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_split=0.2)

train_datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.5,
    horizontal_flip=True,
    vertical_flip=True
)

save_to_dir="./卷积神经网络/pic/"
train_generator=train_datagen.flow(x_train,y_train,batch_size=32,save_to_dir=save_to_dir)
history=model.fit(train_generator,batch_size=32,epochs=5)





