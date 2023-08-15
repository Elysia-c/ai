import tensorflow as tf
def VGGNet(input_shape,padding):
    # 建立Sequential模型
    model=tf.keras.Sequential()
    #卷积层1-2 池化层
    model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding=padding,strides=4,activation=tf.nn.relu,input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3),padding=padding,strides=4,activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),padding=padding))

    #卷积层3-4 池化层
    model.add(tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding=padding,strides=4,activation=tf.nn.relu))
    model.add(tf.keras.layers.Conv2D(128,kernel_size=(3,3),padding=padding,strides=4,activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),padding=padding))

    #卷积层5-7 池化层
    model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),padding=padding))

    #卷积层8-10 池化层
    model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),padding=padding))

    #卷积层11-13 池化层
    model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.Conv2D(512,kernel_size=(3,3),strides=1,padding=padding,activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),padding=padding))
        
    #Flatten层
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
model=VGGNet((224,224,3),"same")
model.summary()