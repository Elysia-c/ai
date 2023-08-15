import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams["axes.unicode_minus"]=False

n=1024
x1=np.random.normal(0,1,n)
y1=np.random.normal(0,1,n)

x2=np.random.uniform(-4,4,(1,n))
y2=np.random.uniform(-4,4,(1,n))

plt.scatter(x1,y1,color="b",marker="*",label="正态分布")
plt.scatter(x2,y2,color="yellow",marker="o",label="均匀分布")

plt.legend()
plt.title("标准正态分布",fontsize=20)
# plt.text(2.5,2.5,"均值：0\n标准差：1")

plt.xlim(-4,4)
plt.ylim(-4,4)

plt.xlabel("横坐标x",fontsize=14)
plt.ylabel("纵坐标y",fontsize=14)

plt.show()
