
import matplotlib.pyplot as plt

# plt.plot()
plt.rcParams["font.sans-serif"]="SimHei"

fig=plt.figure(facecolor="lightgrey")

plt.subplot(2,2,1)
plt.title("子标题1")
plt.subplot(2,2,2)
plt.title("子标题2",loc="left",color="b")
plt.subplot(2,2,3)
plt.title("子标题4",color="white",backgroundcolor="black")
plt.subplot(2,2,4)

plt.suptitle("全局标题",color="red",fontsize=20,backgroundcolor="yellow")

plt.tight_layout(rect=(0,0,1,0.9))

plt.show()