import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

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
train_x=iris_train[:,2:3]
train_y=iris_train[:,4]

test_x=iris_test[:,2:3]
test_y=iris_test[:,4]

# 提取山鸢尾和变色鸢尾
x_train=train_x[train_y<2]
y_train=train_y[train_y<2]

x_test=test_x[test_y<2]
y_test=test_y[test_y<2]

num_train=len(x_train)
num_test=len(x_test)

cm_pt=mpl.colors.ListedColormap(["blue","red"])

x_train=x_train-np.mean(x_train)
x_test=x_test-np.mean(x_test)


plt.figure(figsize=(10,3))

plt.subplot(121)
plt.scatter(x_train,y_train,cmap=cm_pt)

plt.subplot(122)
plt.scatter(x_test,y_test,cmap=cm_pt)

plt.show()

# 设置超参数 学习率
lean_rate=0.2
# 设置超参数 迭代次数
iter=120
# 每1次迭代输出1次
display_step=30

# 设置模型参数初值 w0, b0
np.random.seed(612)
w=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())

x_=range(-80,80)
y_=1/(1+tf.exp(-(w*x_+b)))

plt.scatter(x_train,y_train)
plt.plot(x_,y_,color="red",linewidth=3)
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)

#记录交叉熵损失
cross_train=[]
#记录分类准确率
acc_train=[]
#记录交叉熵损失
cross_test=[]
#记录分类准确率
acc_test=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #计算预测值和均方误差
        pred_train=1/(1+tf.exp(-(w*x_train+b)))
        Loss_train=-tf.reduce_mean(y_train*tf.math.log(pred_train)+(1-y_train)*tf.math.log(1-pred_train))
        Accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train<0.5,0,1),y_train),tf.float32))

        pred_test=1/(1+tf.exp(-(w*x_test+b)))
        Loss_test=-tf.reduce_mean(y_test*tf.math.log(pred_test)+(1-y_test)*tf.math.log(1-pred_test))
        Accuracy_test=tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_test<0.5,0,1),y_test),tf.float32))

    cross_train.append(Loss_train)
    acc_train.append(Accuracy_train)
    cross_train.append(Loss_test)
    acc_train.append(Accuracy_test)

    #计算损失函数对w,b的梯度
    dL_dw,dL_db=tape.gradient(Loss_train,[w,b])
    #更新w和b
    w.assign_sub(lean_rate*dL_dw)
    b.assign_sub(lean_rate*dL_db)

    if i%display_step==0:
        print("i: %i,train Loss: %f,train Accuracy: %f,test Loss: %f,test Accuracy: %f"%(i,Loss_train,Accuracy_train,Loss_test,Accuracy_test))
        y_=1/(1+tf.exp(-(w*x_+b)))
        plt.plot(x_,y_)

plt.show()
plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(cross_train,color="blue",label="train")
plt.plot(cross_test,color="red",label="acc")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
plt.plot(acc_train,color="blue",label="train")
plt.plot(acc_test,color="red",label="acc")
plt.ylabel("Accuracy")

plt.legend()
plt.show()