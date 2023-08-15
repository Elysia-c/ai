import tensorflow as tf
print("Tensorflow version",tf.__version__)
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus, cpus)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#需要多少就使用多少
tf.config.experimental.set_memory_growth(gpus[0],True)

#加载数据
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1],TEST_URL)

df_iris_train=pd.read_csv(train_path,header=0)
df_iris_test=pd.read_csv(test_path,header=0)

# 转化为NumPy数组
iris_train=np.array(df_iris_train)
iris_test=np.array(df_iris_test)

# 提取属性和标签
x_train=iris_train[:,0:4]
y_train=iris_train[:,4]

x_test=iris_test[:,0:4]
y_test=iris_test[:,4]

x_train=x_train-np.mean(x_train,axis=0)
x_test=x_test-np.mean(x_test,axis=0)

X_test=tf.cast(x_test,tf.float32)
#转换为独热编码
Y_test=tf.one_hot(tf.constant(y_test,dtype=tf.int32),3)

X_train=tf.cast(x_train,tf.float32)
#转换为独热编码
Y_train=tf.one_hot(tf.constant(y_train,dtype=tf.int32),3)

# 设置超参数 学习率
lean_rate=0.5
# 设置超参数 迭代次数
iter=65
# 每1次迭代输出1次
display_step=5

# 设置模型参数初值 W,B
np.random.seed(612)
W1=tf.Variable(np.random.randn(4,16),dtype=tf.float32)
B1=tf.Variable(np.zeros([16]),dtype=tf.float32)
W2=tf.Variable(np.random.randn(16,3),dtype=tf.float32)
B2=tf.Variable(np.zeros([3]),dtype=tf.float32)

#记录交叉熵损失
cce_train=[]
cce_test=[]
#记录分类准确率
acc_train=[]
acc_test=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #隐含层输出
        Hidden_train=tf.nn.relu(tf.matmul(X_train,W1)+B1)
        #输出层输出
        PRED_train=tf.nn.softmax(tf.matmul(Hidden_train,W2)+B2)
        Loss_train=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_train,y_pred=PRED_train))

        Hidden_test=tf.nn.relu(tf.matmul(X_test,W1)+B1)
        PRED_test=tf.nn.softmax(tf.matmul(Hidden_test,W2)+B2)
        Loss_test=tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=Y_test,y_pred=PRED_test))

    accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(),axis=1),y_train),tf.float32))
    accuracy_test=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_test.numpy(),axis=1),y_test),tf.float32))

    cce_train.append(Loss_train)
    acc_train.append(accuracy_train)
    cce_test.append(Loss_test)
    acc_test.append(accuracy_test)

    greds=tape.gradient(Loss_train,[W1,B1,W2,B2])
    W1.assign_sub(lean_rate*greds[0])
    B1.assign_sub(lean_rate*greds[1])
    W2.assign_sub(lean_rate*greds[2])
    B2.assign_sub(lean_rate*greds[3])

    if i%display_step==0:
        print("i: %i,TarinLoss: %f,TarinAcc: %f,TestLoss: %f,TestAcc: %f"%(i,Loss_train,accuracy_train,Loss_test,accuracy_test))

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(cce_train,color="blue",label="train")
plt.plot(cce_test,color="red",label="test")
plt.ylabel("Loss")
plt.xlabel("Iteration")
plt.legend()

plt.subplot(122)
plt.plot(acc_train,color="blue",label="train")
plt.plot(acc_test,color="red",label="test")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
