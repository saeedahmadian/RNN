import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from config import *


class RNN(object):

    def __init__(self, num_in, num_out, dropout):
        self.model = Sequential()
        self.model.add(LSTM(units=50,return_sequences=True, input_shape=(num_in,1)))
        self.model.add(Dropout(dropout))
        self.model.add(LSTM(units=30, return_sequences=True))
        self.model.add(Dropout(dropout))
        self.model.add(LSTM(units=20))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(units=num_out))
        self.model.compile(optimizer='adam',loss='mean_squared_error')

    def fit_model(self,x_train, y_train, epochs, batch_size):
        print('Training Process starts...\n')
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)


    def predict_time_series(self,x_test):
        self.model.predict(x_test)


data = normalize(read_file('Google_Stock_Price_Train.csv'))
x , y = create_data_set(30,5,data.shape[0],data)

myRNN =RNN(x.shape[1],y.shape[1],0.3)
myRNN.fit_model(x,y,20,10)



a=1