import pandas as pd
import os

path ='archive (1)'
fname = 'Indicators.csv'

df = pd.read_csv(os.path.join(path, fname))
for c in df.columns:
    print(c)
    print(len(df[c].unique()))
    if c =='IndicatorName':
        for i in df[c].unique():
            print(i)