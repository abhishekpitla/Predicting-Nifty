# Predicting Nifty50 2018 prices using 2017 prices

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



training_set = pd.read_csv('nifty50_2017.csv')
training_set = training_set.iloc[:,1:2].values

# Feature Scaling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
x_train = training_set[0:247]
y_train = training_set[1:248]

# Reshaping the array

x_train = np.reshape(x_train,(247,1,1))


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

reg = Sequential()

# Creating LSTM Layer

reg.add(LSTM(units = 4,activation = 'sigmoid',input_shape  = (None,1)))

# Making the Neural Network

reg.add(Dense(units = 1))

reg.compile(optimizer = 'adam',loss = 'mean_squared_error')

reg.fit(x_train , y_train , batch_size = 32 , epochs = 1000)

test_set = pd.read_csv('nifty50_2018.csv')
real_stock_price = test_set.iloc[:,1:2].values

# Predicting 2018 prices

inputs = real_stock_price
inputs = sc.transform(inputs)

inputs = np.reshape(inputs ,(248,1,1))
predicted_stock_price = reg.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Ploting 

plt.plot(real_stock_price , color = 'red' , label = 'real' )
plt.plot(predicted_stock_price , color = 'blue' , label = 'predicted')
plt.xlabel('Time')
plt.ylabel('nifty')
plt.legend()
plt.show()
