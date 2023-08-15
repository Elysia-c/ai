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
cifar10=tf.keras.datasets.cifar10
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#数据预处理
x_train,x_test=tf.cast(x_train,dtype=tf.float32)/255.0,tf.cast(x_test,dtype=tf.float32)/255.0
y_train,y_test=tf.cast(y_train,tf.int16),tf.cast(y_test,tf.int16)

# #建立Sequential模型
model=tf.keras.Sequential()
#特征提取层1
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding="same",activation=tf.nn.relu,input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding="same",activation=tf.nn.relu))#卷积层
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))#最大池化层
model.add(tf.keras.layers.Dropout(0.2))#随机丢取20%的隐含层节点

#特征提取层2
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding="same",activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding="same",activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.2))

#全连接层
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

model.summary()

#配置训练方法
model.compile(optimizer='adam',# 优化器
              loss='sparse_categorical_crossentropy',# 损失函数
              metrics=['sparse_categorical_accuracy']# 准确值
              )

history=model.fit(x_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

model.evaluate(x_test,y_test,verbose=2)

model.save('./卷积神经网络/CIFAR10_CNN_weight.h5')

# model=tf.keras.models.load_model('./人工神经网络/mnist_model.h5')

print(history.history)
loss=history.history['loss']
acc=history.history['sparse_categorical_accuracy']
val_loss=history.history['val_loss']
val_acc=history.history['val_sparse_categorical_accuracy']

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(loss,color='b',label='train')
plt.plot(val_loss,color='r',label='test')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(acc,color='b',label='train')
plt.plot(val_acc,color='r',label='test')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

plt.figure()
for i in range(10):
    num=np.random.randint(1,10000)

    plt.subplot(2,5,i+1)
    plt.axis("off")
    plt.imshow(x_test[num],cmap="gray")
    demo = tf.reshape(x_test[num],(1,32,32,3))		
    y_pred = np.argmax(model.predict(demo))	#预测样本
    title="标签值"+str((y_test.numpy())[num,0])+"\n预测值="+str(y_pred)
    plt.title(title)

plt.show()