# python
# x=[137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21]
# y=[145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30]
# meanX=sum(x)/len(x)
# meanY=sum(y)/len(y)
# sunXY=0.0
# sunX=0.0
# for i in range(len(x)):
#     sunXY+=(x[i]-meanX)*(y[i]-meanY)
#     sunX+=(x[i]-meanX)*(x[i]-meanX)

# numpy
# import numpy as np
# x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
# y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
# meanX=np.mean(x)
# meanY=np.mean(y)
# sunXY=np.sum((x-meanX)*(y-meanY))
# sunX=np.sum((x-meanX)*(x-meanX))

# numpy
import tensorflow as tf
x=tf.constant([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y=tf.constant([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])
meanX=tf.reduce_mean(x)
meanY=tf.reduce_mean(y)
sunXY=tf.reduce_sum((x-meanX)*(y-meanY))
sunX=tf.reduce_sum((x-meanX)*(x-meanX))

w=sunXY/sunX
b=meanY-w*meanX
# print("w=",w)
# print("b=",b)
print("w=",w.numpy())
print("b=",b.numpy())
x_test=[128.15,45.00,141.43,106.27,99.00,53.84,85.36,70.00]
# print("面积\t估计房价")
# for i in range(len(x_test)):
#     # print(x_test[i],"\t",round(w*x_test[i]+b,2))
#     print(x_test[i],"\t",np.round(w*x_test[i]+b,2))

y_pred=w*x_test+b
print(y_pred)