import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

boston_hosing=tf.keras.datasets.boston_housing
(train_x,train_y),(_,_)=boston_hosing.load_data(test_split=0)

plt.rcParams["font.sans-serif"]="SimHei"
plt.rcParams["axes.unicode_minus"]=False

titles = ["CRIM","ZN","INDUs","CHAS","NOX","RM","AGE","DIs","RAD","TAx","PTRATI0","B-100o" ,"LSTAT","MEDv"]


plt.figure(figsize=(12,12))
for i in range(13):
    plt.subplot(4,4,(i+1))
    plt.scatter(train_x[:,i],train_y)
    plt.xlabel(titles[i])
    plt.ylabel("Price($1000's)")
    plt.title(str(i+1)+"."+titles[i]+"-Price")

plt.tight_layout()
plt.suptitle("各属性与房价的关系",x=0.5,y=1.02,fontsize=20)
plt.show()