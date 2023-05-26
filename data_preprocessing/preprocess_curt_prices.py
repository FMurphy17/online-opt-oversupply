from pandasql import sqldf
import pandas as pd
from sklearn import datasets
import os  
import datetime

df = pd.read_csv('../CAISO_data/caiso_daily_prices.csv')
print(df.head())

for i in df.Date:
    i = i.replace('/','',1)
    i = i.replace('/','20')
    i = datetime.datetime.strptime(i, "%m%d%Y").date()
df['Date']= pd.to_datetime(df['Date'])

#df.info()
#print(df.head())

dingdong = pd.DataFrame()

all_dates = set(df['Date'])
for each_date in sorted(all_dates):
    prices = list(df[df.Date == each_date]['Price'])
    #avgs = []
    maxs = []
    for i in range(0, 288, 12):
        p = prices[i:i+12]
        try:
            #a = sum(p) / len(p)
            b = max(p)
            dingdong = pd.concat([dingdong,
                                  pd.DataFrame([[each_date, b]],
                                              columns=['Date', 'Price'])],
                                ignore_index=True)
        except:
            pass


os.makedirs('../CAISO_data/', exist_ok=True)  
dingdong.to_csv('../CAISO_data/caiso_avg_prices_24.csv', index=False)  