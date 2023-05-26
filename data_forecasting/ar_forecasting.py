from pandasql import sqldf
from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import lag_plot
from pandas import DataFrame
from pandas import concat
import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
import math
import csv
from datetime import date, timedelta, datetime
import os
from statsmodels.tsa.ar_model import AutoReg
from math import sqrt

"""
This tutorial: https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
was used to construct an AR forecasting model for the data forecasting step of the online optimization model.
Refer to the tutorial for a more in-depth look into AR forecasting!
"""


# read in desired dataset for data prediction 
# see data folder for data sets that were used for experiements & input formatting of data
df = pd.read_csv('../BTC_data/btc_hrate_pp.csv', header=0, index_col=0)
#print(df)

# modify select statement to create different experiments
q = "select Date, Value from df where Date < '2021-06-01' order by Date"
series = sqldf(q, globals()).set_index('Date')
#print(series)
series.plot() 
pyplot.show()

# lag plot
lag_plot(series)
pyplot.show()
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print("Pearson correlation")
result = dataframe.corr()
print(result)

# autocorrelation plot
autocorrelation_plot(series)
pyplot.show()

# acf plot
plot_acf(series, lags=70)
pyplot.show()


# AR model
# 7-day forecast at hourly resolution == 168 data forecast

# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-6:]
# train autoregression

# lags = single param to be tuned for accuracy 
model = AutoReg(train, lags=40)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
print("length of predictions:", len(predictions))
print("length of test:", len(test))
print("predictions type: ", type(predictions))

for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    #print('predicted=%f' % (predictions[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()


# save future data so that it can be loaded into the RHC algorithm
future_data = model_fit.predict(start=len(X), end=len(X) + 7, dynamic=False)

# change lines based on experiment 
os.makedirs('./experiments/exp1/rhc/', exist_ok=True)
with open('./experiments/exp1/rhc/btc_hr.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Date", "Value"])
    # change line below based on starting date of experiment 
    start_date = date(2021, 6, 1)
    j = 1
    for i in future_data:
        # make date in desired format which becomes first column 
        # insert the predicted value from future_data into second column
        writer.writerow([start_date, i])
        # uncomment the line below if the forecasted data is not at an hourly resolution (i.e.
        # BTC difficulty/hashrate/closing price which is replicated 24 times for each day's calculations)
        #if j%24==0:
        start_date = start_date + timedelta(1)
        j += 1
