# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:25:12 2018

@author: Joish
"""
import numpy as np
import tensorflow as tf
import random as rn
    
import os
os.environ['PYTHONHASHSEED'] = '0'
    
np.random.seed(44)
    
rn.seed(12345)
    
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    
from keras import backend as K
    
tf.set_random_seed(1234)
    
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('stock.csv')
train_set = dataset.iloc[:, 1:2].values

scaler = MinMaxScaler(feature_range = (0, 1))
train_set_scaled = scaler.fit_transform(train_set)

Xtrain = train_set_scaled[0:1257]
Ytrain = train_set_scaled[1:1258]

Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model

model = Sequential()

model.add(LSTM(units =250, return_sequences = True, input_shape = (Xtrain.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units =250))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

from datetime import datetime
start=datetime.now()

model.fit(Xtrain, Ytrain, epochs = 50, batch_size = 32)

time = datetime.now()-start

#model.save('my_model_05.h5')

test = pd.read_csv('Test.csv')
testX = test.iloc[:,1:2].values

inputs = testX
inputs = scaler.transform(inputs)
inputs = np.reshape(inputs,(inputs.shape[0],1,1))

predS  = model.predict(inputs)
pred = scaler.inverse_transform(predS)

plt.plot(testX,color = 'red',label='real price')
plt.plot(pred,color = 'blue',label='Predicted price')
plt.title('Prediction')
plt.xlabel('time')
plt.ylabel('Stock price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error
import math

error = math.sqrt(mean_squared_error(testX,pred))
percent = error/(sum(testX)/testX.shape[0])
percent = percent*100

print('-------------------------------------------')
print ('RMSE:'+str(error))
print ('Percentage:'+str(percent[0])+' %')
print('-------------------------------------------')
print ('Time for Training:'+str(time))
print('-------------------------------------------')

xplot = []
yrplot=[]
ypplot=[]
for i in range(pred.shape[0]):
    xplot.append(int(i))
    yrplot.append(int(testX[i]))
    ypplot.append(int(pred[i]))

import plotly.plotly as py
from plotly.graph_objs import Scatter, Figure, Layout
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

trace1 = Scatter(
    x=xplot,
    y=yrplot,
    name = 'Real Stock Price',
    
)

trace2 = Scatter(
    x=xplot,
    y=ypplot,
    name = 'Predicted Stock Price'
)

plot({
      'data': [trace1,trace2], 
      'layout': {
                  'title': 'VISUAL RESULTS', 
                  'font': dict(size=16)
                }
     }
    )