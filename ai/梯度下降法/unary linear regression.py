
import matplotlib.pyplot as plt
import numpy as np

# 数据
x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

# 设置超参数 学习率
lean_rate=0.00001
# 设置超参数 迭代次数
iter=100
# 每十次迭代输出1次
display_step=10

# 设置模型参数初值 w0, b0
np.random.seed(612)
w=np.random.randn()
b=np.random.randn()

#每次迭代的损失值
mse=[]

for i in range(0,iter+1):
    #计算损失函数对w的偏导数
    dL_dw=np.mean(x*(w*x+b-y))
     #计算损失函数对b的偏导数
    dL_db=np.mean(x*(w*x+b-y))

    #使用迭代公式更新w
    w=w-lean_rate*dL_dw
    #使用迭代公式更新b
    b=b-lean_rate*dL_db

    #算估计值
    pred=w*x+b
    #计算均分误差
    Loss=np.mean(np.squeeze(y-pred)/2)
    mse.append(Loss)

    plt.plot(x,pred)

    if i%display_step==0:
        print("i: %i,Loss: %s,w: %f,b: %f"%(i,mse[i],w,b))

plt.rcParams["font.sans-serif"]="SimHei"

plt.figure()
plt.scatter(x,y,color="red",label="销售记录")
plt.scatter(x,pred,color="blue",label="梯度下降法")
plt.plot(x,pred,color="blue")
plt.plot(x,0.89*x+5.41,color="green")

plt.xlabel("Area",fontsize=14)
plt.ylabel("Price",fontsize=14)

plt.figure()

plt.plot(mse)

plt.xlabel("Iteration",fontsize=14)
plt.ylabel("Loss",fontsize=14)

plt.legend(loc="upper left")
plt.show()