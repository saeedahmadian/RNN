import pandas as pd
import numpy as np
import csv
from datetime import datetime

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
gas_col = list(map(lambda i : 'GAS_'+i,columns[3:9]))
temp_col = list(map(lambda i : 'Temp_'+i,columns[9:]))
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

# for ind,element in enumerate(columns,start=0):
#     if ind < 2 and ind <9:
#         new_col.append('GAS_'+element)
#     elif ind > 8:
#         new_col.append('Temp_' + element)

a=1
