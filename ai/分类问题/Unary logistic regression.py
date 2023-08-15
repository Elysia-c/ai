import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

x = np.array([137.97,104.50,100.00,126.32,79.20,99.00,124.00,114.00,106.69,140.05,53.75,46.91,68.00,63.02,81.26,86.21])
y = np.array([1,1,0,1,0,1,1,0,0, 1,0,0,0,0,0,0])

#点中心化
x_train=x-np.mean(x)
y_train=y

# 设置超参数 学习率
lean_rate=0.005
# 设置超参数 迭代次数
iter=5
# 每1次迭代输出1次
display_step=1

# 设置模型参数初值 w0, b0
np.random.seed(612)
w=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())

x_=range(-80,80)
y_=1/(1+tf.exp(-(w*x_+b)))

plt.scatter(x_train,y_train)
plt.plot(x_,y_,color="red",linewidth=3)

#记录交叉熵损失
cross_train=[]
#记录分类准确率
acc_train=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #计算预测值和均方误差
        pred_train=1/(1+tf.exp(-(w*x_train+b)))
        Loss_train=-tf.reduce_mean(y_train*tf.math.log(pred_train)+(1-y_train)*tf.math.log(1-pred_train))
        Accuracy_train=tf.reduce_mean(tf.cast(tf.equal(tf.where(pred_train<0.5,0,1),y_train),tf.float32))

    cross_train.append(Loss_train)
    acc_train.append(Accuracy_train)

    #计算损失函数对w,b的梯度
    dL_dw,dL_db=tape.gradient(Loss_train,[w,b])
    #更新w和b
    w.assign_sub(lean_rate*dL_dw)
    b.assign_sub(lean_rate*dL_db)

    if i%display_step==0:
        print("i: %i,Train Loss: %f,Accuracy: %f"%(i,Loss_train,Accuracy_train))
        y_=1/(1+tf.exp(-(w*x_+b)))
        plt.plot(x_,y_)
        

plt.show()
x_test=[128.15,45.00,141.43,106.27,99.00,53.84,85.36,70.00,162.00,114.60]
pred_test=1/(1+tf.exp(-(w*(x_test-np.mean(x))+b)))
y_test=tf.where(pred_test<0.5,0,1)
for i in range(len(x_test)):
    print(x_test[i],"\t",pred_test[i].numpy(),"\t",y_test[i].numpy(),"\t")

plt.scatter(x_test,y_test)
x_=range(-80,80)
y_=1/(1+tf.exp(-(w*x_+b)))
plt.plot(x_+np.mean(x),y_)
plt.show()