# -*- coding: utf-8 -*- 
# @Time : 2022/10/27 10:29 
# @Author : YeMeng 
# @File : demo1.py 
# @contact: 876720687@qq.com

"""
tokenizer = Tokenizer(num_words=500, split=' ')
tokenizer.fit_on_texts(data_v1['verified_reviews'].values)
X = tokenizer.texts_to_sequences(data['verified_reviews'].values)
X = pad_sequences(X)
"""
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from sklearn.model_selection import train_test_split
from tensorflow.python.estimator import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dense, Dropout
pd.set_option('display.max_columns',None) # 显示所有列
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./input/train.csv')
data_test = pd.read_csv('./input/test.csv')
df0 = pd.read_csv('./input/sample_submission.csv')
data_test = data_test['full_text']

# -------------------- 数据向量化 -----------------------
X_train=df['full_text']
y_train=np.array(df.iloc[:,2:])
tokenizer = Tokenizer(num_words=10000, split=' ')
tokenizer.fit_on_texts(X_train.values)
X_train = tokenizer.texts_to_sequences(X_train.values)
data_test = tokenizer.texts_to_sequences(data_test.values)
X_train = pad_sequences(X_train) #
data_test = pad_sequences(data_test)
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1)

# -------------------- 构建模型 -----------------------
model=Sequential()
model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 5000, and
# output embedding dimension of size 120.
model.add(Embedding(input_dim=10000, output_dim=120))
# Add a LSTM layer with 128 internal units.
model.add(LSTM(128))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='relu'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='relu'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='relu'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='relu'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='linear'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='linear'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='linear'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='linear'))
# Add a Dense layer with 128 units and activation relu.
model.add(Dense(128,activation='linear'))
# Add a Dropout layer with persent 57% out units.
model.add(Dropout(0.57))
# Add a Dense layer with 1 units and activation relu.
model.add(Dense(6,activation='linear'))
#create compile with loss mean_sequard_error and optmizer adam.
model.compile(
    loss = 'mse',
    optimizer ='adam'
)

model.fit(X_train, y_train, epochs = 57)

val_loss=model.evaluate(X_val,y_val)

y_pr=model.predict(data_test)

df0.to_csv('submission.csv',index=False)