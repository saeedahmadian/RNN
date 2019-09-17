import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA

data = pd.read_csv('Google_Stock_Price_Train.csv',index_col='Date',header=0)
data.index = pd.to_datetime(data.index)
data_grad_1 = data.iloc[:,0:2].diff().dropna()
data_grad_2 = data_grad_1.iloc[:,0:2].diff().dropna()

# To obtain the number of autoregresive terms AR (anything outside of the the shady env is statistically significant)
plot_pacf(data_grad_2.iloc[:,0])
plt.show()
#to obtain the number of moving average terms MA ((anything outside of the the shady env is statistically significant))
plot_acf(data_grad_2.iloc[:,0])
plt.show()

a=1