import tensorflow as tf
print("Tensorflow version",tf.__version__)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

from matplotlib import pyplot as plt
import numpy as np

#加载数据
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()

#数据预处理
X_train,X_test=tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)
y_train,y_test=tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

#建立Sequential模型
model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))

# model=tf.keras.models.load_model('./人工神经网络/mnist_model.h5')

model.summary()

#配置训练方法
model.compile(optimizer='adam',# 优化器
              loss='sparse_categorical_crossentropy',# 损失函数
              metrics=['sparse_categorical_accuracy']# 准确值
              )

model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)


# 保存模型
model.save_weights("./人工神经网络/mnist_weights.h5")

# #加载模型
# model.load_weights("./人工神经网络/mnist_weights.h5")

model.evaluate(X_test,y_test,verbose=2)

for i in range(4):
    num=np.random.randint(1,10000)

    plt.subplot(1,4,i+1)
    plt.axis("off")
    plt.imshow(test_x[num],cmap="gray")
    demo = tf.reshape(X_test[num],(1,28,28))		#增加数组维度，将维度变为(1,28,28)
    y_pred = np.argmax(model.predict(demo))	#预测样本
    title="y="+str(test_y[num])+"\ny_pred="+str(y_pred)
    plt.title(title)

plt.show()

# model.save("./人工神经网络/mnist_model.h5")