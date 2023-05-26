import pandas as pd
from datetime import date, timedelta, datetime
import math
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

"""
This is the code for the offline optimization algorithm. The algorithm uses multiple historical
data sets and will assess a week's worth of data at an hourly resolution in an online fashion
to determine how much would-be curtailed power should be distributed to BTC mining/battery storage such that
profit is maximized. This is a benchmark for the RHC algorithm as this algorithm assumes perfect
knowledge of the future by treating historical data as the 'forecasted' data. 
""" 

def vectorize(df):
    ovec = []
    #print(df.head())
    for idx, x in enumerate(df.iloc[:, 0]):
        if idx >= 8: # 7 for oo, 8 for rhc
            break
        vec = []
        for i in range(24):
            vec.append(x)
        ovec.append(vec)
    return ovec

def daterange(start_date, end_date):
    delta = timedelta(hours=1)
    while start_date < end_date:
        yield start_date
        start_date += delta

def vectorize2(df):
    ovec = []
    #print(df.head())
    currentDate = df.iloc[0]['Date']
    inner_lst = 0
    ovec.append([])
    for idx, x in enumerate(df.iloc[:, 1]):
        if df.iloc[idx]['Date'] == currentDate:
            ovec[inner_lst].append(x)
        else: 
            currentDate = df.iloc[idx]['Date']
            ovec.append([])
            inner_lst += 1
            ovec[inner_lst].append(x)
    return ovec

def vectorFlatten(vector2d):
    result = []
    for i in vector2d:
        for j in i:
            result.append(j)
    return result


# Data Loading
def data_load(price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log):
    # Note: some var names say 'logged' but i ended up not logging anything as this was messing with
    # subsequent calculations so ignore

    price_btc_r = pd.read_csv(price_btc, header=0, index_col=0)
    price_btc_2d = vectorize(price_btc_r)
    price_btc = np.array(vectorFlatten(price_btc_2d)[:168])

    hash_t_r = pd.read_csv(hash_t, header=0, index_col=0)
    hash_t_2d = vectorize(hash_t_r)
    hash_t = np.array(vectorFlatten(hash_t_2d)[:168])

    diff_r = pd.read_csv(diff, header=0, index_col=0)
    diff_2d = vectorize(diff_r)
    diff = np.array(vectorFlatten(diff_2d)[:168])

    # avg price of electricity each day (time instant t) in CAISO
    price_energy_r = pd.read_csv(price_energy_log, header=0)
    price_energy = vectorize2(price_energy_r)
    price_energy_log = np.array(vectorFlatten(price_energy)[:168])
    

    # note that max price was found by averaging the top 60 prices each day (max 5 hours of prices)
    max_price_energy_r = pd.read_csv(max_price_energy, header=0, index_col=0)
    max_price_energy_2d = vectorize(max_price_energy_r)
    max_price_energy = np.array(vectorFlatten(max_price_energy_2d)[:168])

    CP_r = pd.read_csv(CP_log, header=0) # NONLOGGED
    CP = vectorize2(CP_r)
    CP_log = np.array(vectorFlatten(CP)[:168])
    
    
    return price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log


def offline_optimal(price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log, startdate, enddate):
    #print("OO")
    data_values = data_load(price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log)
    price_btc = data_values[0]
    hash_t = data_values[1]
    diff = data_values[2]
    price_energy_log = data_values[3]
    max_price_energy = data_values[4]
    CP_log = data_values[5]
    reward_post20 = 6.25
    const1 = reward_post20 / 2**32

    # STEP 1: Initialize state 
    charges = []
    charges.append(0)
    #batt_chrg_0 = np.zeros(168)
    batt_chrg = cp.Variable((168,))

    #profit_1 = 0
    #profit_2 = 0
    x_bm = cp.Variable(168) # intialize as a vector
    x_pm = cp.Variable(168)
    x_bs = cp.Variable(168)
    x_pb = cp.Variable(168)
    i_chrg = cp.Variable(168)
    ic_matrix = cp.diag(i_chrg)
    
    i_dischrg = cp.Variable(168)
    idc_matrix = cp.diag(i_dischrg)

    bc = 2607 # this is MW in total across all of CAISO
    dis_rate = 120
    chrg_rate = 120
        
    # STEP 2: Loop over total, T, timeframe; for offline optimal we are just going to assess the entire timeframe at once
    # STEP 3: Observe current state (i.e. p1, p2, batt_charge)
    # STEP 4: Compute Optimal Control Sequence - this is the convex optimization problem

    time_v1 = (price_btc * hash_t) / diff
    fbtc = const1 * time_v1

    charges.append(batt_chrg[-1] + x_pb[-1] - x_bm[-1] - x_bs[-1])
    if x_pb.value is not None and x_bm.value is not None and x_bs.value is not None and batt_chrg.value is not None:
        batt_chrg[1:] = batt_chrg[:-1] + x_pb[:-1] - x_bm[:-1] - x_bs[:-1]
    
    
    # cvxpy
    M = 1000 
    constraints = [np.zeros(168) <= x_bm, np.zeros(168) <= x_bs, np.zeros(168) <= x_pm,
                   np.zeros(168) <= x_pb, x_pb <= np.ones(168) * chrg_rate,
                   x_bm + x_bs <= cp.minimum(dis_rate, batt_chrg[-1]), x_bm + x_bs <= batt_chrg,
                   x_pm + x_pb <= CP_log, 
                   
                   batt_chrg[:-1] + x_pb[:-1] <= np.ones(167) * bc,
                   batt_chrg[1:] == batt_chrg[:-1] + x_pb[:-1] - x_bm[:-1] - x_bs[:-1],
                   batt_chrg[0] == 15] 
    
    obj_m = (x_bm + x_pm) @ fbtc - (x_pm @ price_energy_log) 
    obj_b = (x_bs @ max_price_energy) - (x_pb @ price_energy_log)
    obj_fx = cp.Maximize(obj_m + obj_b)
    prob = cp.Problem(obj_fx, constraints)
    
    prob.solve()
    
    #STEP 5: For offline optimal, we get the actions and max profits directly from the optimization problem that we solved in step 4
    print("----Actions----")
#     print("Power to mining from battery: ", x_bm.value)
#     print("Power sold from battery: ", x_bs.value)
#     print("Power to mining: ", x_pm.value)
#     print("Power to battery: ",x_pb.value)
    
    #print("Battery charge", batt_chrg.value)
    #print("Curtailed power", CP_log)
    #print("i_chrg", i_chrg.value)
    #print("i_dischrg", i_dischrg.value)


    #print("Battery Charge:", (batt_charge).value)
    print("\n")
    print("----Objective Values----")
    print("Optimal Value (Total Profit): ", prob.value)
    
    # how should i be breaking down profit from battery cycled back to mining bc this is kind of both methods contributing to profit
    print("Profit from Mining: ", obj_m.value)
    print("Profit from Battery: ", obj_b.value)

    df = pd.DataFrame()
    hr_res = []
    start_date = startdate
    end_date = enddate
    for single_date in daterange(start_date, end_date):
        hr_res.append(single_date)
        #.strftime("%Y-%m-%d %H:%M")
    
    alldates = []
    for i in range(168):
        alldates.append(hr_res[i])
    for i in range(168):
        alldates.append(hr_res[i])
    for i in range(168):
        alldates.append(hr_res[i])
    for i in range(168):
        alldates.append(hr_res[i])
    for i in range(168):
        alldates.append(hr_res[i])
    
    column_type = []
    for i in range(840):
        if i < 168:
            column_type.append("Curtailed Power")
        elif i >= 168 and i < 336:
            column_type.append('Power to Mining')
        elif i >= 336 and i < 504:
            column_type.append('Power to Batteries')
        elif i >= 504 and i < 672:
            column_type.append('Power to Mining from Batteries')
        else:
            column_type.append('Power Sold from Batteries')
   
    mydata = []
    for i in range(168):
        mydata.append(CP_log[i])
    for i in range(168):
        mydata.append(x_pm.value[i])
    for i in range(168):
        mydata.append(x_pb.value[i])
    for i in range(168):
        mydata.append(x_bm.value[i])
    for i in range(168):
        mydata.append(x_bs.value[i])
        
    df['Time'] = alldates
    df["Power Destination"] = column_type 
    df['Power (mW/h)'] = mydata
    fig = px.line(df, x="Time", y="Power (mW/h)", color='Power Destination')
    fig.show()

    return prob.value, obj_m.value, obj_b.value


# Example invocation: 

# oo2 = offline_optimal('./experiments/exp2/oo/btc_prices.csv', './experiments/exp2/oo/btc_hr.csv',
#                 './experiments/exp2/oo/btc_diff.csv', './experiments/exp2/oo/caiso_avg_prices.csv', 
#                 './experiments/exp2/oo/caiso_max_prices.csv','./experiments/exp2/oo/curtailed_power.csv')

# where the return values are 

# total_profit --> oo2[0]
# mining_profit --> oo2[1]
# battery_profit --> oo2[2]
