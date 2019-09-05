import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = data.iloc[:,1:2].values



sc= MinMaxScaler(feature_range=(0,1))

training_set_norm = sc.fit_transform(training_set)

X_train = []
Y_train = []

for i in range(60,1258):
    X_train.append(training_set_norm[i-60:i,0])
    Y_train.append(training_set_norm[i,0])

x_train ,y_train = np.array(X_train), np.array(Y_train)

x_train =np.reshape(x_train, (x_train.shape[0],x_train.shape[0],1))
