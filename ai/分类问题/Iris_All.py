import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

#加载数据
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)

df_iris_train=pd.read_csv(train_path,header=0)


# 转化为NumPy数组
iris_train=np.array(df_iris_train)

# 提取属性和标签
x_train=iris_train[:,2:4]
y_train=iris_train[:,4]

num_train=len(x_train)

cm_pt=mpl.colors.ListedColormap(["blue","red"])

x_train=x_train-np.mean(x_train,axis=0)

x0_train=np.ones(num_train).reshape(-1,1)
X_train=tf.cast(tf.concat([x0_train,x_train],axis=1),tf.float32)
#转换为独热编码
Y_train=tf.one_hot(tf.constant(y_train,dtype=tf.int32),3)


# 设置超参数 学习率
lean_rate=0.2
# 设置超参数 迭代次数
iter=500
# 每1次迭代输出1次
display_step=100

# 设置模型参数初值 W
np.random.seed(612)
W=tf.Variable(np.random.randn(3,3),dtype=tf.float32)

#记录交叉熵损失
ce=[]
#记录分类准确率
acc=[]

for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        #计算预测值和均方误差
        PRED_train=tf.nn.softmax(tf.matmul(X_train,W))
        Loss=-tf.reduce_sum(Y_train*tf.math.log(PRED_train))/num_train
        
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(PRED_train.numpy(),axis=1),y_train),tf.float32))
    
    ce.append(Loss)
    acc.append(accuracy)

    #计算损失函数对w的梯度
    dL_dW=tape.gradient(Loss,W)
    #更新w
    W.assign_sub(lean_rate*dL_dW)

    if i%display_step==0:
        print("i: %i,Loss: %f,Acc: %f"%(i,Loss,accuracy))


plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(ce,color="blue",label="Loss")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
plt.plot(acc,color="blue",label="accuracy")
plt.ylabel("Accuracy")

plt.legend()
plt.show()

#转换为自然顺序码
tf.argmax(PRED_train.numpy(),axis=1)

M=500
#根据属性花瓣长度和花瓣宽度的最大值和最小值
x1_min,x2_min=x_train.min(axis=0)
x1_max,x2_max=x_train.max(axis=0)
#生成网格点坐标矩阵
t1=np.linspace(x1_min,x1_max,M)
t2=np.linspace(x2_min,x2_max,M)
m1,m2=np.meshgrid(t1,t2)

m0=np.ones(M*M)
X_=tf.cast(np.stack((m0,m1.reshape(-1),m2.reshape(-1)),axis=1),tf.float32)
Y_=tf.nn.softmax(tf.matmul(X_,W))

Y_=tf.argmax(Y_.numpy(),axis=1)

n=tf.reshape(Y_,m1.shape)

plt.figure(figsize=(8,6))

cm_bg=mpl.colors.ListedColormap(["#a0ffa0","#ffa0a0","#a0a0ff"])

plt.pcolormesh(m1,m2,n,cmap=cm_bg)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap="brg")

plt.show()