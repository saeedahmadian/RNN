import pandas as pd
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose as decomp
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
# df = pd.read_csv('Saeed.csv',header=3, delimiter=";")

with open('Saeed.csv') as csvfile:
    reader = csv.reader(csvfile)
    DateIndex = []
    Gas_press = []
    Gas_temp =[]
    for row in reader:
        if row[0]== 'DATE' :
            columns= row
        if '1397' in row[0]:
            DateIndex.append(row[0])
            Gas_press.append(list(map(float, map(lambda row : row + '0' if row=='' or row=='-' else row,row[3:]))))
        elif 'INLET' in row[0]:
            print(row[3:])
            Gas_temp.append(list(map(float, map(lambda row : row + '0' if row=='' or row=='-' else row,row[2:8]))))



columns += columns[3:]
gas_col = list(map(lambda i : 'GAS_Pressure'+i,columns[3:9]))
temp_col = list(map(lambda i : 'GAS_Temperature_'+i,columns[9:]))
col = gas_col+temp_col
# col.insert(0,columns[0])

data = np.concatenate((np.array(Gas_press).reshape([-1,len(Gas_press[0])]),np.array(Gas_temp).reshape([-1,len(Gas_temp[0])])),axis=1)
DateIndex=np.array(DateIndex).reshape([-1,1])
date =[]
# for counter,dt in enumerate(DateIndex[:,0],0):
#     print('counter = {} and date = {}'.format(counter,dt))
#     tmp= datetime.strptime(dt,'%Y/%m/%d')
#     date.append(tmp)
#     print(datetime.strptime(dt,'%Y/%m/%d'))


# date = [datetime.strptime(i,'%Y/%m/%d') for i in DateIndex]
df= pd.DataFrame(data= data, columns= col)
dateind = pd.date_range(start='21/4/2018', end='21/06/2018')
df.index =dateind
df.replace(0,np.nan).to_csv('raw_data.csv')
df = df.replace(0,np.nan).interpolate('linear',0)
df_new_MA_press = df - df.rolling(7).mean()
df_new_EWM_press = df - df.ewm(halflife=7,ignore_na=False,min_periods=0).mean()
df_first = df.diff(1)
df_second = df.diff(2)
df_log = np.log(df)
df_log_MA = df_log- df_log.rolling(7).mean()
# for ind,element in enumerate(columns,start=0):
#     if ind < 2 and ind <9:
#         new_col.append('GAS_'+element)
#     elif ind > 8:
#         new_col.append('Temp_' + element)

df1 = df.shift(1).dropna()
df2 = df.shift(2).dropna()
df1 = df1.iloc[1:,:]
df1.columns = [col+'_lag1' for col in df1.columns]
df2.columns = [col+'_lag2' for col in df2.columns]
df_new = df.iloc[2:,:]
data = pd.concat([df1,df2,df_new],axis=1)
X = data[['GAS_Temperature_OT-1_lag1','GAS_Temperature_OT-1_lag2']]
X= sm.add_constant(X)
Y = df_new.iloc[:,6]
mdl = sm.OLS(Y,X)
res = mdl.fit()


def Stationarytest(TS):
    test = adfuller(TS, autolag='AIC')
    result = pd.Series(test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in test[4].items():
        result['Critical Value (%s)' % key] = value
    return result
def Decomposition(TS, nameOfTS, model_name='additive',frequency=None):
    decomposed= decomp(x=TS,model=model_name, freq= frequency)
    out= pd.DataFrame()
    out['Original_'+nameOfTS] = TS
    out["trend"] = decomposed.trend
    out["Seasonality"] = decomposed.seasonal
    out["residual"]= decomposed.resid
    return out

def plot_decomposition(Decomp_TS):
    """
    """
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8),sharex=True)

    ax1.plot(Decomp_TS.iloc[:,0], label=Decomp_TS.columns[0], color='black')
    ax1.legend(loc='best')
    ax1.tick_params(axis='x', rotation=45)

    ax2.plot(Decomp_TS.iloc[:,1], label='Trend Element', color='magenta')
    ax2.legend(loc='best')
    ax2.tick_params(axis='x', rotation=45)

    ax3.plot(Decomp_TS.iloc[:,2], label='Seasonality Element', color='green')
    ax3.legend(loc='best')
    ax3.tick_params(axis='x', rotation=45)

    ax4.plot(Decomp_TS.iloc[:,3], label='Residuals Element', color='red')
    ax4.legend(loc='best')
    ax4.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    TSname=Decomp_TS.columns[0]
    # Show graph
    # plt.suptitle('Trend, Seasonal, and Residual Decomposition of %s' % (TSname),
    #              x=0.5,
    #              y=1.05,
    #              fontsize=18)
    # plt.show()
    plt.savefig(TSname + '.png')
    plt.close()

def TimeSerieOLS(TS,lag):
    TS_lag = TS.shift(lag)
    TS_lag = TS_lag.dropna()
    TS = TS.iloc[lag:,:]
    res = []
    newdata=[]
    for col in TS.columns:
        res.append(sm.OLS(TS[col], TS_lag[col]).fit())
        tmp = pd.DataFrame(data=sm.OLS(TS[col], TS_lag[col]).fit().predict(), index=TS[col].index,columns=[col+'_prediction'])
        tmp[col]=TS[col]
        tmp[col +'_prediction_error'] = tmp[col]-tmp[col + '_prediction']
        newdata.append(tmp)
    return res,newdata


def ErrorOLS(err,lag):
    err=pd.DataFrame(data=err,columns=[err.name])
    err_lag = err.shift(lag).dropna()
    err = err.iloc[lag:,:]
    model = []
    res = []
    for col in err.columns:
        model.append(sm.OLS(err[col],err_lag[col]))
        res.append(sm.OLS(err[col],err_lag[col]).fit())
    return res

a=1