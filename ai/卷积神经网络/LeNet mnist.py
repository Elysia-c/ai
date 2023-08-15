import tensorflow as tf
print("Tensorflow version",tf.__version__)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras import layers,Sequential

#加载数据
plt.rcParams["font.sans-serif"]="SimHei"
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()

#数据预处理
X_train,X_test=tf.cast(train_x,tf.float32)/255.0,tf.cast(test_x,tf.float32)/255.0
y_train,y_test=tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

X_train=train_x.reshape(60000,28,28,1)
X_test=test_x.reshape(10000,28,28,1)

#建立Sequential模型
model=tf.keras.Sequential()
#卷积层1 最大池化层1
model.add(tf.keras.layers.Conv2D(filters=6,kernel_size=5,padding="same",activation="sigmoid",input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#卷积层2 最大池化层2
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=5,padding="valid",activation="sigmoid"))
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
              loss='sparse_categorical_crossentropy',# 损失函数
              metrics=['accuracy']# 准确值
              )

history=model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

model.evaluate(X_train,y_train,verbose=1)
model.evaluate(X_test,y_test,verbose=1)

model.save_weights('./卷积神经网络/mnist_weights.h5')

# model=tf.keras.models.load_model('./人工神经网络/mnist_model.h5')

# print(history.history)
# loss=history.history['loss']
# acc=history.history['sparse_categorical_accuracy']
# val_loss=history.history['val_loss']
# val_acc=history.history['val_sparse_categorical_accuracy']

# plt.figure(figsize=(10,3))

# plt.subplot(121)
# plt.plot(loss,color='b',label='train')
# plt.plot(val_loss,color='r',label='test')
# plt.ylabel('loss')
# plt.legend()

# plt.subplot(122)
# plt.plot(acc,color='b',label='train')
# plt.plot(val_acc,color='r',label='test')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()

# plt.figure()
# for i in range(10):
#     num=np.random.randint(1,10000)

#     plt.subplot(2,5,i+1)
#     plt.axis("off")
#     plt.imshow(x_test[num],cmap="gray")
#     demo = tf.reshape(x_test[num],(1,32,32,3))		
#     y_pred = np.argmax(model.predict(demo))	#预测样本
#     title="标签值"+str((y_test.numpy())[num,0])+"\n预测值="+str(y_pred)
#     plt.title(title)

# plt.show()