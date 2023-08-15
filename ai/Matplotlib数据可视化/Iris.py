import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1],TRAIN_URL)
COLUMN_NAMES = ['SepalLength','Sepalwidth' , 'PetalLength' , 'Petalwidth', 'Species ']
df_iris=pd.read_csv(train_path,names=COLUMN_NAMES,header=0)
print(df_iris.head())
iris=np.array(df_iris)