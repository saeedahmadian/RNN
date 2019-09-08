import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def read_file(name):
    return pd.read_csv(name).iloc[:,1:2].values


def normalize(data):
    norm = MinMaxScaler(feature_range=(0,1))
    return norm.fit_transform(data)


def create_data_set(num_feature,output_size,sample_size, time_series):
    x_train = []
    y_train =[]
    for i in range(num_feature,sample_size-output_size):
        x_train.append(time_series[i-num_feature:i,0])
        y_train.append(time_series[i:i+output_size,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    return x_train, y_train