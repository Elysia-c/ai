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
train_x=iris_train[:,0:2]
train_y=iris_train[:,4]

test_x=iris_test[:,0:2]
test_y=iris_test[:,4]

# 提取山鸢尾和变色鸢尾
x_train=train_x[train_y<2]
y_train=train_y[train_y<2]

x_test=test_x[test_y<2]
y_test=test_y[test_y<2]

num_train=len(x_train)
num_test=len(x_test)

cm_pt=mpl.colors.ListedColormap(["blue","red"])

x_train=x_train-np.mean(x_train,axis=0)
x_test=x_test-np.mean(x_test,axis=0)


plt.figure(figsize=(10,3))

plt.subplot(121)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)

plt.subplot(122)
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap=cm_pt)

plt.show()


x0_train=np.ones(num_train).reshape(-1,1)
X_train=tf.cast(tf.concat((x0_train,x_train),axis=1),tf.float32)
Y_train=tf.cast(y_train.reshape(-1,1),tf.float32)

x0_test=np.ones(num_test).reshape(-1,1)
X_test=tf.cast(tf.concat((x0_test,x_test),axis=1),tf.float32)
Y_test=tf.cast(y_test.reshape(-1,1),tf.float32)

# 设置超参数 学习率
lean_rate=0.2
# 设置超参数 迭代次数
iter=120
# 每1次迭代输出1次
display_step=30

# 设置模型参数初值 W
np.random.seed(612)
W=tf.Variable(np.random.randn(3,1),dtype=tf.float32)
x_=[-1.5,1.5]
y_=-(W[1]*x_+W[0])/W[2]

plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=cm_pt)
plt.plot(x_,y_,color="r",linewidth=3)
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

#记录交叉熵损失
ce_train=[]
ce_test=[]
#记录分类准确率
acc_train=[]
acc_test=[]


for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #计算预测值和均方误差
        PRED_train=1/(1+tf.exp(-tf.matmul(X_train,W)))
        Loss_train=-tf.reduce_mean(Y_train*tf.math.log(PRED_train)+(1-Y_train)*tf.math.log(1-PRED_train))
        PRED_test=1/(1+tf.exp(-tf.matmul(X_test,W)))
        Loss_test=-tf.reduce_mean(Y_test*tf.math.log(PRED_test)+(1-Y_test)*tf.math.log(1-PRED_test))
        
    accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_train.numpy()<0.5,0.,1.),Y_train),tf.float32))
    accuracy_test=tf.reduce_mean(tf.cast(tf.equal(tf.where(PRED_test.numpy()<0.5,0.,1.),Y_test),tf.float32))
    
    ce_train.append(Loss_train)
    acc_train.append(accuracy_train)
    ce_test.append(Loss_test)
    acc_test.append(accuracy_test)

    #计算损失函数对w的梯度
    dL_dW=tape.gradient(Loss_train,W)
    #更新w
    W.assign_sub(lean_rate*dL_dW)

    if i%display_step==0:
        print("i: %i,TrainLoss: %f,TrainAcc: %f,TestLoss: %f,TestAcc: %f"%(i,Loss_train,accuracy_train,Loss_test,accuracy_test))
        y_=-(W[1]*x_+W[0])/W[2]
        plt.plot(x_,y_)


plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(ce_train,color="blue",label="train")
plt.plot(ce_test,color="red",label="acc")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
plt.plot(acc_train,color="blue",label="train")
plt.plot(acc_test,color="red",label="acc")
plt.ylabel("Accuracy")

plt.legend()
plt.show()

