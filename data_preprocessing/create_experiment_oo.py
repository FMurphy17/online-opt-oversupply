import pandas as pd
from datetime import date, datetime, timedelta
from pandasql import sqldf
import os  

price_btc = pd.read_csv('../BTC_data/btc_price_pp.csv', header=0, index_col=0)

hashr = pd.read_csv('../BTC_data/btc_hrate_pp.csv', header=0, index_col=0)

diff = pd.read_csv('../BTC_data/btc_diff_pp.csv', header=0, index_col=0)

avg_prices_e = pd.read_csv('../CAISO_data/caiso_avg_prices_24.csv', header=0, index_col=0)

max_prices_e = pd.read_csv('../CAISO_data/caiso_max_price_pp.csv', header=0, index_col=0)

cp = pd.read_csv('../CAISO_data/curtailment_24_pp.csv', header=0, index_col=0)

def create_exp(startdate, enddate, expnum):
    q1 = "select Date, Price from avg_prices_e where Date >= '" + startdate + "' and Date < '" + enddate + "' ORDER BY Date"
    newdf1 = sqldf(q1, globals())
    os.makedirs('./experiments/exp' + expnum + '/oo', exist_ok=True)  
    newdf1.to_csv('./experiments/exp' + expnum + '/oo/caiso_avg_prices.csv', index=False)  
    
    q2 = "select Date, Price from max_prices_e where Date >= '" + startdate + "' and Date < '" + enddate + "' ORDER BY Date"
    newdf2 = sqldf(q2, globals())
    os.makedirs('./experiments/exp' + expnum + '/oo', exist_ok=True)  
    newdf2.to_csv('./experiments/exp' + expnum + '/oo/caiso_max_prices.csv', index=False) 

    q3 = "select Date, Curtailment from cp where Date >= '" + startdate + "' and Date < '" + enddate + "' ORDER BY Date"
    newdf3 = sqldf(q3, globals())
    os.makedirs('./experiments/exp' + expnum + '/oo', exist_ok=True)  
    newdf3.to_csv('./experiments/exp' + expnum + '/oo/curtailed_power.csv', index=False) 

    q4 = "select Date, Value from diff where Date >= '" + startdate + "' and Date < '" + enddate + "' ORDER BY Date"
    newdf4 = sqldf(q4, globals())
    os.makedirs('./experiments/exp' + expnum + '/oo', exist_ok=True)  
    newdf4.to_csv('./experiments/exp' + expnum + '/oo/btc_diff.csv', index=False) 

    q5 = "select Date, Value from hashr where Date >= '" + startdate + "' and Date < '" + enddate + "' ORDER BY Date"
    newdf5 = sqldf(q5, globals())
    os.makedirs('./experiments/exp' + expnum + '/oo', exist_ok=True)  
    newdf5.to_csv('./experiments/exp' + expnum + '/oo/btc_hr.csv', index=False) 

    q6 = "select Date, Close from price_btc where Date >= '" + startdate + "' and Date < '" + enddate + "' ORDER BY Date"
    newdf6 = sqldf(q6, globals())
    os.makedirs('./experiments/exp' + expnum + '/oo', exist_ok=True)  
    newdf6.to_csv('./experiments/exp' + expnum + '/oo/btc_prices.csv', index=False)
    
# create_exp("2021-06-01", "2021-06-08", "1")
# create_exp("2021-12-01", "2021-12-08", "2")
# create_exp("2022-03-01", "2022-03-08", "3")
# create_exp("2022-07-01", "2022-07-08", "4")
# create_exp("2022-11-01", "2022-11-08", "5")