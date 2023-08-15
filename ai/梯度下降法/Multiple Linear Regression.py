import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"]="SimHei"

# 数据
area=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
room=np.array([3,2,2,3,1,2,3,2,2,3,1,1,1,1,2,2])
price=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
num=len(area)

x0=np.ones(num)

#线性归一化
x1=(area-area.min())/(area.max()-area.min())
x2=(room-room.min())/(room.max()-room.min())

X=np.stack((x0,x1,x2),axis=1)
Y=price.reshape(-1,1)

# 设置超参数 学习率
lean_rate=0.001
# 设置超参数 迭代次数
iter=500
# 每50次迭代输出1次
display_step=50

# 设置模型参数初值 w0,
np.random.seed(612)
W=np.random.randn(3,1)

#每次迭代的损失值
mse=[]

for i in range(0,iter+1):
    #计算偏导数
    dL_dW=np.matmul(np.transpose(X),np.matmul(X,W)-Y)
    #更新权值
    W=W-lean_rate*dL_dW

    #算估计值
    PRED=np.matmul(X,W)
    #计算均分误差
    Loss=np.mean(np.squeeze(Y-PRED)/2)
    mse.append(Loss)

    if i%display_step==0:
        print("i: %i,Loss: %f"%(i,mse[i]))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(mse)
plt.xlabel("Iteration",fontsize=14)
plt.ylabel("Loss",fontsize=14)

plt.subplot(1,2,2)
PRED=PRED.reshape(-1)
plt.plot(price,color="red",marker="o",label="销售记录")
plt.plot(PRED,color="blue",marker=".",label="梯度下降法")
plt.xlabel("Sample",fontsize=14)
plt.ylabel("Price",fontsize=14)

plt.legend()
plt.show()