import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

boston_hosing=tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=boston_hosing.load_data()

num_train=len(train_x)
num_test=len(test_x)

#数据归一化
x_train=(train_x-train_x.min(axis=0))/(train_x.max(axis=0)-train_x.min(axis=0))
y_train=train_y

x_test=(test_x-test_x.min(axis=0))/(test_x.max(axis=0)-test_x.min(axis=0))
y_test=test_y

x0_train=np.ones(num_train).reshape(-1,1)
x0_test=np.ones(num_test).reshape(-1,1)

#生成多元回归需要的二维数组的形式
X_train=tf.cast(tf.concat([x0_train,x_train],axis=1),tf.float32)
X_test=tf.cast(tf.concat([x0_test,x_test],axis=1),tf.float32)

Y_train=tf.constant(y_train.reshape(-1,1),tf.float32)
Y_test=tf.constant(y_test.reshape(-1,1),tf.float32)

# 设置超参数 学习率
lean_rate=0.01
# 设置超参数 迭代次数
iter=2800
# 每200次迭代输出1次
display_step=200

# 设置模型参数初值 
np.random.seed(612)
W=tf.Variable(np.random.randn(14,1),dtype=tf.float32)

#记录误差
mse_train=[]
mse_test=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #计算预测值和均方误差
        PRED_train=tf.matmul(X_train,W)
        Loss_train=0.5*tf.reduce_mean(tf.square(Y_train-PRED_train))
        
        PRED_test=tf.matmul(X_test,W)
        Loss_test=0.5*tf.reduce_mean(tf.square(Y_test-PRED_test))

    mse_train.append(Loss_train)
    mse_test.append(Loss_train)

    #计算损失函数对w,b的梯度
    dL_dW=tape.gradient(Loss_train,W)
    #更新w和b
    W.assign_sub(lean_rate*dL_dW)

    if i%display_step==0:
        print("i: %i,Train Loss: %f,Test Loss: %f"%(i,Loss_train,Loss_test))

plt.figure(figsize=(20,4))

plt.subplot(131)
plt.ylabel("MSE")
#训练集和测试集损失值变化应一致，避免过拟合。在出现点应及时停止迭代
plt.plot(mse_train,color="blue",linewidth=3,label="train loss")
plt.plot(mse_test,color="red",linewidth=1.5,label="test loss")
plt.legend(loc="upper right")

plt.subplot(132)
plt.plot(y_train,color="blue",marker="o",label="ture_price")
plt.plot(PRED_train,color="red",marker=".",label="predict")
plt.legend()

plt.subplot(133)
plt.plot(y_test,color="blue",marker="o",label="ture_price")
plt.plot(PRED_test,color="red",marker=".",label="predict")
plt.legend()

plt.show()