import pandas as pd
from datetime import date, timedelta, datetime
import math
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

"""
This is the code for the RHC (online optimization) algorithm. The algorithm uses multiple 
forecasted data sets and will assess a week's worth of data at an hourly resolution in an online fashion
to determine how much would-be curtailed power should be distributed to BTC mining/battery storage such that
profit is maximized. 
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

"""
This routine is used to calculate the total profit of the grid operator computed by the RHC algorithm. 
"""
def calculate_cost(x_pb, x_pm, x_bs, x_bm, price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log):
    # load the actual data using data_load
    data_values = data_load(price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log)
    price_btc = data_values[0]
    hash_t = data_values[1]
    diff = data_values[2]
    price_energy_log = data_values[3]
    max_price_energy = data_values[4]
    CP_log = data_values[5]
    
    # change action lists to np arrays
    arr_x_pm = np.array(x_pm)
    arr_x_pb = np.array(x_pb)
    arr_x_bm = np.array(x_bm)
    arr_x_bs = np.array(x_bs)
    
    reward_post20 = 6.25
    const1 = reward_post20 / 2**32
    
    time_v1 = (price_btc * hash_t) / diff
    fbtc = const1 * time_v1
    
    profit_m = np.dot((arr_x_bm + arr_x_pm), fbtc) - np.dot(arr_x_pm, price_energy_log) 
    profit_b = np.dot(arr_x_bs, max_price_energy) - np.dot(arr_x_pb, price_energy_log) 
    total_profit = profit_m + profit_b
    return total_profit, profit_m, profit_b
    # solve objective function using loaded data and the decision variable params

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

def pred_data_load(price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log):
    # Note: some var names say 'logged' but i ended up not logging anything as this was messing with
    # subsequent calculations so ignore

    price_btc_r = pd.read_csv(price_btc, header=0, index_col=0)
    price_btc_2d = vectorize(price_btc_r)
    price_btc = np.array(vectorFlatten(price_btc_2d)[:173])

    hash_t_r = pd.read_csv(hash_t, header=0, index_col=0)
    hash_t_2d = vectorize(hash_t_r)
    hash_t = np.array(vectorFlatten(hash_t_2d)[:173])

    diff_r = pd.read_csv(diff, header=0, index_col=0)
    diff_2d = vectorize(diff_r)
    diff = np.array(vectorFlatten(diff_2d)[:173])

    # avg price of electricity each day (time instant t) in CAISO
    price_energy_r = pd.read_csv(price_energy_log, header=0)
    price_energy = vectorize2(price_energy_r)
    price_energy_log = np.array(vectorFlatten(price_energy)[:173])

    # note that max price was found by averaging the top 60 prices each day (max 5 hours of prices)
    max_price_energy_r = pd.read_csv(max_price_energy, header=0, index_col=0)
    max_price_energy_2d = vectorize(max_price_energy_r)

    # yes i made a whole different routine just to change this array indexing, yes i am lazy
    max_price_energy = np.array(vectorFlatten(max_price_energy_2d)[:173])

    CP_r = pd.read_csv(CP_log, header=0) 
    CP = vectorize2(CP_r)
    CP_log = np.array(vectorFlatten(CP)[:173])
    
    return price_btc, hash_t, diff, price_energy_log, max_price_energy, CP_log


def rhc(price_btc_p, hash_t_p, diff_p, price_energy_log_p, max_price_energy_p, CP_log_p,
        price_btc_a, hash_t_a, diff_a, price_energy_log_a, max_price_energy_a, CP_log_a, startdate, enddate):    
    print("RHC")
    # predicted data
    data_values = pred_data_load(price_btc_p, hash_t_p, diff_p, price_energy_log_p, max_price_energy_p, CP_log_p)
    price_btc_p = data_values[0]
    hash_t_p = data_values[1]
    diff_p = data_values[2]
    price_energy_log_p = data_values[3]
    max_price_energy_p = data_values[4]
    CP_log_p = data_values[5]

    # actual data
    data_values_a = data_load(price_btc_a, hash_t_a, diff_a, price_energy_log_a, max_price_energy_a, CP_log_a)
    price_btc_a = data_values_a[0]
    hash_t_a = data_values_a[1]
    diff_a = data_values_a[2]
    price_energy_log_a = data_values_a[3]
    max_price_energy_a = data_values_a[4]
    CP_log_a = data_values_a[5]
    
    formatted = format_preds(price_btc_p, hash_t_p, diff_p, price_energy_log_p, max_price_energy_p, CP_log_p,
                price_btc_a, hash_t_a, diff_a, price_energy_log_a, max_price_energy_a, CP_log_a)
    price_btc = formatted[0]
    hash_t = formatted[1]
    diff = formatted[2]
    price_energy_log = formatted[3]
    max_price_energy = formatted[4]
    CP_log = formatted[5]
    
    reward_post20 = 6.25
    const1 = reward_post20 / 2**32
    
    # STEP 1: Initialize state 
    charges = []
    batt_chrg = cp.Variable((6,))
    profit_m = 0
    profit_b = 0
    bc = 2607 # this is MW in total across all of CAISO
    dis_rate = 120 # technically this should be max(10, 105% of local capacity need)
    chrg_rate = 120
    
    #cvxpy setup
    x_bm = cp.Variable(6)
    x_pm = cp.Variable(6)
    x_bs = cp.Variable(6)
    x_pb = cp.Variable(6)
    
    x_bm_action = []
    x_pm_action = []
    x_bs_action = []
    x_pb_action = []
    
    # STEP 2: Loop over total, T, timeframe
    # T = 7 days
    t = 0
    w = 0
    ts = []
    for i in range(7): # 7 days in a week 
        # w = 6 hrs
        for j in range(24): # 24 hrs in a day --> final result = 24x7=168-d 
            # STEP 3: Observe current state (i.e. profit_m, profit_b, batt_charge)
            # STEP 4: Compute Optimal Control Sequence - this is the convex optimization problem

            time_v1 = (price_btc[w:w+6] * hash_t[w:w+6]) / diff[w:w+6]
            fbtc = const1 * time_v1

            batt_chrg = batt_chrg + x_pb - x_bm - x_bs
            if batt_chrg[0].value == None:
                # BEWARE! This can be finnicky if the initial battery charge is too low...i do not know why!
                charges.append(15)
            else:
                charges.append(batt_chrg[0].value)
            
            # cvxpy
            M = 1000 # big-M constant
            constraints = [np.zeros(6) <= x_bm, np.zeros(6) <= x_bs, np.zeros(6) <= x_pm, np.zeros(6) <= x_pb,
                            x_pb <= np.ones(6) * chrg_rate, x_bm + x_bs <= cp.minimum(dis_rate, batt_chrg[-1]), 
                            x_bm + x_bs <= batt_chrg, x_pm + x_pb <= CP_log[w:w+6], 
                            batt_chrg[:-1] + x_pb[:-1] <= np.ones(5) * bc,
                            batt_chrg[1:] == batt_chrg[:-1] + x_pb[:-1] - x_bm[:-1] - x_bs[:-1],
                            batt_chrg[0] == charges[-1]]  

            obj_m = ((x_bm + x_pm) @ fbtc) - (x_pm @ price_energy_log[w:w+6])
            obj_b = (x_bs @ max_price_energy[w:w+6]) - (x_pb @ price_energy_log[w:w+6]) 
            obj_fx = cp.Maximize(obj_m + obj_b)
            prob = cp.Problem(obj_fx, constraints)
            prob.solve()
            # print()

            #STEP 5: For offline optimal, we get the actions and max profits directly from the optimization problem that we solved in step 4
#             print("Iteration", t)
#             print("----Actions----")
#             print("Power to mining from battery: ", x_bm.value)
#             print("Power sold from battery: ", x_bs.value)
#             print("Power to mining: ", x_pm.value)
#             print("Power to battery: ", x_pb.value)
#             print("Battery charge", batt_chrg.value)
        
            if x_pb.value is not None and x_bm.value is not None and x_bs.value is not None and x_pm.value is not None:
                x_bm_action.append(x_bm.value[0])
                x_pm_action.append(x_pm.value[0])
                x_bs_action.append(x_bs.value[0])
                x_pb_action.append(x_pb.value[0])
                #print("Battery charge", batt_chrg.value)
                profit_m += obj_m.value
                profit_b += obj_b.value
            else:
                x_bm_action.append(0)
                x_pm_action.append(0)
                x_bs_action.append(0)
                x_pb_action.append(0)
                #print("Battery charge", batt_chrg.value)
                profit_m += 0
                profit_b += 0
#             print("\n")
#             print("----Objective Values----")
#             print("Optimal Value (Total Profit): ", prob.value)

#             print("Profit from Mining: ", obj_m.value)
            #print("Profit from Battery: ", obj_b.value)
            #print("\n")
            
            t += 1
            w += 1
            #print("Profit from mining after window", j, ":", profit_m)
            #print("Profit from battery after window", j, ":", profit_b)

    print("---Final action lists---")
#     print("Power to battery actions: ", x_pb_action, "\n")
#     print("Power to mining actions: ", x_pm_action, "\n")
#     print("Power sold from battery actions: ", x_bs_action, "\n")
#     print("Power from battery to mining actions: ", x_bm_action, "\n") 
#     print("Battery charges: ", charges)

    tp = calculate_cost(x_pb_action, x_pm_action, x_bs_action, x_bm_action, './experiments/exp1/oo/btc_prices.csv', './experiments/exp1/oo/btc_hr.csv',
                './experiments/exp1/oo/btc_diff.csv', './experiments/exp1/oo/caiso_avg_prices.csv', 
                './experiments/exp1/oo/caiso_max_prices.csv','./experiments/exp1/oo/curtailed_power.csv')
    print("Profit from battery: ", tp[2])
    print("Profit from mining: ", tp[1])
    print("Total Profit: ", tp[0])
    
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
        mydata.append(x_pm_action[i])
    for i in range(168):
        mydata.append(x_pb_action[i])
    for i in range(168):
        mydata.append(x_bm_action[i])
    for i in range(168):
        mydata.append(x_bs_action[i])
        
    df['Time'] = alldates
    df["Power Destination"] = column_type 
    df['Power (mW/h)'] = mydata
    fig = px.line(df, x="Time", y="Power (mW/h)", color='Power Destination')
    fig.show()
    
    return tp


# Example invocation: 

# tp2 = rhc('./experiments/exp2/rhc/btc_prices.csv', './experiments/exp2/rhc/btc_hr.csv',
#                 './experiments/exp2/rhc/btc_diff.csv', './experiments/exp2/rhc/caiso_avg_prices.csv', 
#                 './experiments/exp2/rhc/caiso_max_prices.csv','./experiments/exp2/rhc/curtailed_power.csv',
#                 './experiments/exp2/oo/btc_prices.csv', './experiments/exp2/oo/btc_hr.csv',
#                 './experiments/exp2/oo/btc_diff.csv', './experiments/exp2/oo/caiso_avg_prices.csv', 
#                 './experiments/exp2/oo/caiso_max_prices.csv','./experiments/exp2/oo/curtailed_power.csv')

# where the return values are 

# total_profit --> tp2[0]
# mining_profit --> tp2[1]
# battery_profit --> tp2[2]
