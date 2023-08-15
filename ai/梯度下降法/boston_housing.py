import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

boston_hosing=tf.keras.datasets.boston_housing
(train_x,train_y),(test_x,test_y)=boston_hosing.load_data()

#取出房间数
x_train=train_x[:,5]
y_train=train_y

x_test=test_x[:,5]
y_test=test_y

# 设置超参数 学习率
lean_rate=0.04
# 设置超参数 迭代次数
iter=2000
# 每200次迭代输出1次
display_step=200

# 设置模型参数初值 w0, b0
np.random.seed(612)
w=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())

#记录误差
mse_train=[]
mse_test=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #计算预测值和均方误差
        pred_train=w*x_train+b
        loss_train=0.5*tf.reduce_mean(tf.square(y_train-pred_train))
        
        pred_test=w*x_test+b
        loss_test=0.5*tf.reduce_mean(tf.square(y_test-pred_test))

    mse_train.append(loss_train)
    mse_test.append(loss_train)

    #计算损失函数对w,b的梯度
    dL_dw,dL_db=tape.gradient(loss_train,[w,b])
    #更新w和b
    w.assign_sub(lean_rate*dL_dw)
    b.assign_sub(lean_rate*dL_db)

    if i%display_step==0:
        print("i: %i,Train Loss: %f,Test Loss: %f"%(i,loss_train,loss_test))

plt.figure(figsize=(15,10))

plt.subplot(221)
plt.scatter(x_train,y_train,color="blue",label="data")
plt.plot(x_train,pred_train,color="red",label="model")
plt.legend(loc="upper left")

plt.subplot(222)
#训练集和测试集损失值变化应一致，避免过拟合。在出现点应及时停止迭代
plt.plot(mse_train,color="blue",linewidth=3,label="train loss")
plt.plot(mse_test,color="red",linewidth=1.5,label="test loss")
plt.legend(loc="upper right")

plt.subplot(223)
plt.plot(y_train,color="blue",marker="o",label="ture_price")
plt.plot(pred_train,color="red",marker=".",label="predict")
plt.legend()

plt.subplot(224)
plt.plot(y_test,color="blue",marker="o",label="ture_price")
plt.plot(pred_test,color="red",marker=".",label="predict")
plt.legend()

plt.show()