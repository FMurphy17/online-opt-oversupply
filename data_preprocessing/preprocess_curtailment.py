from pandasql import sqldf
import pandas as pd
from sklearn import datasets
import os  
import datetime

"""
This is a hode-podge of all the pre-processing I did for every data set I used, which each required
their own pre-processing modifications. So it is most likely illogical and unuseable! :D
"""

#df = pd.read_csv('../BTC_data/Yahoo_Finance/btc_usd_5yr.csv')
df = pd.read_csv('../CAISO_data/Curtailment_Data/curtailment_18_22.csv')
print(df.head())

for i in df.Date:
    i = i.replace('/','',1)
    i = i.replace('/','20')
    i = datetime.datetime.strptime(i, "%m%d%Y").date()
df['Date']= pd.to_datetime(df['Date'])

df.info()
print(df.head())

# df = pd.read_csv('../BTC_data/BCHAIN-HRATE.csv')
# print(df.head())

q = "select Date, Hour, avg(Curtailment) as Curtailment from df group by Date, Hour ORDER BY Date"
newdf = sqldf(q, globals())
print(newdf.head())
# q = "select * from df ORDER BY Date"
# newdf = sqldf(q, globals())

# os.makedirs('../CAISO_data/Curtailment_Data/', exist_ok=True)  
# newdf.to_csv('../CAISO_data/Curtailment_Data/curt_18_22_pp.csv', index=False)  

# need to populate missing hrs for each day (0-23) with 0s for curtailment

# make a newdf_2 with the dates and hours columns precreated
# then just move curtailment col over based on matching dates/hours and put 0s in every other slot
import warnings
warnings.filterwarnings("ignore")
newdf_2 = pd.DataFrame()
all_dates = set(newdf['Date'])
for dt in sorted(all_dates):
    hours = set(newdf[newdf.Date == dt]['Hour'])
    #print(f"{dt} has {hours}")
    for hour in range(0, 24):
        if hour not in hours:
            newdf_2 = pd.concat([newdf_2, pd.DataFrame([[dt, hour, 0.0]],
                                                      columns=['Date', 'Hour', 'Curtailment'])],
                                ignore_index=True)
        else:
            newdf_2 = pd.concat([newdf_2, newdf[newdf.Date == dt][newdf.Hour == hour]], ignore_index=True)

print("XXX ", newdf_2[0:80])


#os.makedirs('../CAISO_data/', exist_ok=True)  
#newdf.to_csv('../CAISO_data/curtailment_24_pp.csv', index=False)  

pd.set_option('display.max_rows', None)
newdf_2.head(80)

os.makedirs('../CAISO_data/', exist_ok=True)  
newdf_2.to_csv('../CAISO_data/curtailment_24_pp.csv', index=False)  