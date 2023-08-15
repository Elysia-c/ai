import tensorflow as tf
def LeNet(input_shape,padding):
    #建立Sequential模型
    model=tf.keras.Sequential()
    #卷积层1 最大池化层1
    model.add(tf.keras.layers.Conv2D(filters=6,kernel_size=5,padding=padding,activation="sigmoid",input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

    #卷积层2 最大池化层2
    model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=5,padding=padding,activation="sigmoid"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

    #卷积层3
    model.add(tf.keras.layers.Conv2D(filters=120,kernel_size=5,padding=padding,activation="sigmoid"))
    model.add(tf.keras.layers.Flatten())

    #全连接层
    model.add(tf.keras.layers.Dense(84,activation="sigmoid"))
    #输出层
    model.add(tf.keras.layers.Dense(10,activation="softmax"))
    
    return model