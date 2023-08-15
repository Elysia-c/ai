import tensorflow as tf
def AlexNet(input_shape,padding):
    # 建立Sequential模型
    model=tf.keras.Sequential()
    #卷积层1 最大池化层1
    model.add(tf.keras.layers.Conv2D(filters=96,kernel_size=(11,11),padding=padding,strides=4,activation="relu",input_shape=input_shape))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding=padding))

    #卷积层2 最大池化层2
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(5,5),padding=padding,strides=1,activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding=padding))

    #卷积层3、4、5 最大池化层3
    model.add(tf.keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=384,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding=padding))
        
    #Flattenc层
    model.add(tf.keras.layers.Flatten())

    #全连接层1 Dropout层
    model.add(tf.keras.layers.Dense(4096,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    #全连接层1 Dropout层
    model.add(tf.keras.layers.Dense(4096,activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))

    #输出层
    model.add(tf.keras.layers.Dense(1000,activation="softmax"))
    
    return model
model=AlexNet((224,224,3),"same")
model.summary()