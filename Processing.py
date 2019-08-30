import pandas as pd
import numpy as np
import csv

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
            Gas_temp.append(list(map(float, map(lambda row : row + '0' if row=='' or row=='-' else row,row[3:]))))





a=1