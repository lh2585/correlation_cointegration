# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:59:08 2018

@author: Lay
"""

import os
import pandas as pd
import numpy as np
import statsmodels
from statsmodels import tsa
import statsmodels.tsa.stattools as ts
from datetime import datetime, timedelta, date, time
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from pandas.tseries.offsets import *
from scipy.stats import skew, kurtosis, kstest

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
mid_path = 'C:/E-books & documents/NYU/2018 Spring/Statistical Arbitrage/Final Project/mid_feb_1mo.csv'

mid = pd.read_csv(mid_path, parse_dates=['Dates'], date_parser=dateparse,header = 0, usecols=["Dates","Mid","Sym"])
mid['date'] = [d.date() for d in mid['Dates']]

# Only take 1, 2, 5 (1560,780,312 training points and 390,195,78 test points - respectively)
frequency = 1
mid = mid[[(int(str(d.time())[-5:-3]) % frequency == 0) for d in mid['Dates']]]

# Taking entry one-sided critical Z-value for 95%
entry = 1.65 
stop = 4.5

# We consider Close (one-sided critical Z-value) vs Frequency (1,2,5)
# Below are the critical Z-values for which we test
# 1.04 <-> 85%, 0.674 <-> 75%, 0.385 <-> 65%, 0.125 <-> 55%, 0 <-> 50%
close = 0.125

training_length = 3
min_correlation = 0.95
min_adf = -3.43
num_of_pairs = 10
leverage = 2
# Note that leverage 2 means long pos + |Short pos| = 2*cash
# So that means that leverage 1 is no leverage, and leverage 0 means no positions.

fixed_tc = 7
good_trade = 0
bad_trade = 0


# Create a folder in your directory
# And it will create a folder with all the results of the runs.
base_dir = 'C:/E-books & documents/NYU/2018 Spring/Statistical Arbitrage/Final Project/Final Result'
_doc = os.path.join(base_dir, 'freq-' + str(frequency) + '-close-' +str(close) +'-%s' %datetime.now().strftime('%Y-%m-%d-%Hh%M'))
os.makedirs(_doc)



# start of stuff
dates = mid.date.unique()
portfolio_results = pd.DataFrame(index = dates[(training_length+1):], columns = ['Portfolio_PNL'])

dateparse2 = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
spx = pd.read_csv(base_dir + '^GSPC.csv',usecols = ['Date','Adj Close'], parse_dates=['Date'], date_parser=dateparse2, index_col = ['Date'])

init_portfolio = 1000*spx['Adj Close'].iloc[0]
portfolio_val = init_portfolio

for i in range(0,len(portfolio_results)):
    start_date = dates[i]
    
    end_date = dates[i+training_length] #n+1 days since it's inclusive, so days = 3 is actually 4 days worth of data
    test_date = dates[i+training_length+1]
    
    _docsubset = os.path.join(_doc, '%s' % test_date.strftime('%Y-%m-%d'))
    os.makedirs(_docsubset)
    
    training_data = mid[(mid.date >= start_date) & (mid.date <= end_date) ]
    training_data = training_data.reset_index(drop=True)
    test_data = mid[mid.date == test_date]
    test_data = test_data.reset_index(drop=True)
    del training_data['date']
    
    corr = training_data.set_index(['Dates','Sym']).unstack().corr()
    corr.columns = corr.columns.droplevel()
    corr.index = corr.columns.tolist()  
    corr.head()
    unstacked_corr = corr.unstack()
    corr_sorted = unstacked_corr.sort_values(kind='quicksort',ascending = False)
    corr_sorted = corr_sorted[(corr_sorted > min_correlation) & (corr_sorted<1)] 
    corr_sorted = corr_sorted[0::2]
    corr_sorted = pd.DataFrame(data=corr_sorted, columns = ['Correlations'])
    corr_sorted['adf'] = ''
    corr_sorted['mu'] = ''
    corr_sorted['beta'] = ''
    corr_sorted['mean_error'] = ''
    corr_sorted['std_error'] = ''
    corr_sorted['total_pnl'] = ''
    
    for j in range(0,len(corr_sorted)):
        tckr1 = corr_sorted.index[j][0]
        tckr2 = corr_sorted.index[j][1]
        
        X =  training_data.loc[(training_data.Sym == tckr1),'Mid'].reset_index(drop=True)
        X = sm.add_constant(X)
        y = training_data.loc[(training_data.Sym == tckr2),'Mid'].reset_index(drop=True)
        model = sm.OLS(y,X).fit()
        corr_sorted.loc[(tckr1, tckr2),'mu'] = model.params[0]
        corr_sorted.loc[(tckr1, tckr2),'beta'] = model.params[1]
        corr_sorted.loc[(tckr1, tckr2),'mean_error'] = model.resid.mean()
        corr_sorted.loc[(tckr1, tckr2),'std_error'] = model.resid.std()
        corr_sorted.loc[(tckr1, tckr2),'adf'] = ts.adfuller(model.resid,autolag = 'BIC')[0]
    
    corr_sorted = corr_sorted.sort_values(by = ['adf'])
    
    corr_sorted = corr_sorted[corr_sorted.adf < min_adf]
    if len(corr_sorted) > num_of_pairs:
        corr_sorted = corr_sorted[:num_of_pairs]
        
    
    index_list = []
    for a in range(0,len(corr_sorted.index)):
        for b in range(0,len(corr_sorted.index[a])):
            index_list.append(corr_sorted.index[a][b])
    
    stock_sum = 0
    for j in range(0,len(index_list)):
        stock_sum += training_data[training_data.Sym == index_list[j]].loc[:,'Mid'].iloc[-1]
    
    pos = ((leverage) * portfolio_val) / stock_sum # Assume one trade per pair per day.
    
    for j in range(0,len(corr_sorted)):
        row = corr_sorted.iloc[j,:]
        tckr1 = corr_sorted.index[j][0]
        tckr2 = corr_sorted.index[j][1]
        error = []
        error_dist = []
        A_pos = 0
        A_close = 0
        A_open = 0
        B_pos = 0
        B_close = 0
        B_open = 0
        trade = False
        pnl = []
        minute_pnl = [0]
        
        for k in range(0,len(test_data.loc[(test_data.Sym == tckr2)])):
            
            pos = ((leverage) * portfolio_val) / stock_sum
            
            P_a = test_data.loc[(test_data.Sym == tckr2),'Mid'].iloc[k]
            P_b = test_data.loc[(test_data.Sym == tckr1),'Mid'].iloc[k]
            error.append(P_a - (row.mu + row.beta * P_b))
            error_dist.append((error[k] - row.mean_error)/row.std_error)
    
            if(k>0):  
                P_am1 = test_data.loc[(test_data.Sym == tckr2),'Mid'].iloc[k-1]
                P_bm1 = test_data.loc[(test_data.Sym == tckr1),'Mid'].iloc[k-1]
                minute_pnl.append((P_a-P_am1)*(A_pos) +(P_b-P_bm1)*(B_pos))
                if(trade):  
                    # Stop-Loss
                    if ((error_dist[k] > stop) | (error_dist[k] < -stop)):
                        A_close = P_a
                        B_close = P_b
                        trade_pnl = (A_close-A_open)*(A_pos) + (B_close-B_open)*(B_pos)
                        if(trade_pnl <0): 
                            bad_trade +=1
                        if(trade_pnl >0): 
                            good_trade +=1
                        print('I hit Stop-Loss at ', test_data.loc[(test_data.Sym == tckr2),'Dates'].iloc[k])
                        A_pos = 0
                        B_pos = 0 
                        trade = False
                        pnl.append(trade_pnl)
                        portfolio_val -= fixed_tc
                        minute_pnl[k] -= fixed_tc
                        portfolio_val += trade_pnl
                        print(portfolio_val)
                        
                    # Close Position
                    if ((error_dist[k] < close) & (error_dist[k] > 0) | (error_dist[k] > -close) & (error_dist[k] < 0)):
                        A_close = P_a
                        B_close = P_b
                        trade_pnl = (A_close-A_open)*(A_pos) +(B_close-B_open)*(B_pos)
                        if(trade_pnl <0): 
                            bad_trade +=1
                        if(trade_pnl >0): 
                            good_trade +=1
                        print('I closed my position at ', test_data.loc[(test_data.Sym == tckr2),'Dates'].iloc[k])
                        A_pos = 0
                        B_pos = 0 
                        trade = False
                        pnl.append(trade_pnl)
                        portfolio_val -= fixed_tc
                        minute_pnl[k] -= fixed_tc
                        portfolio_val += trade_pnl
                        print(portfolio_val)
                        
                    if (k == (len(test_data.loc[(test_data.Sym == tckr2)])-1)):
                        A_close = P_a
                        B_close = P_b
                        trade_pnl = (A_close-A_open)*(A_pos) +(B_close-B_open)*(B_pos)
                        if(trade_pnl <0): 
                            bad_trade +=1
                        if(trade_pnl >0): 
                            good_trade +=1
                        print('It is the end of the day.')
                        A_pos = 0
                        B_pos = 0 
                        trade = False
                        pnl.append(trade_pnl)
                        portfolio_val -= fixed_tc
                        minute_pnl[k] -= fixed_tc
                        portfolio_val += trade_pnl
                        print(portfolio_val)
                        
                # Entry 
                if ((error_dist[k] >= entry) & (error_dist[k-1]<entry) & (~trade) & (test_data.loc[(test_data.Sym == tckr2),'Dates'].iloc[k].time() < time(15,30))): # Entry from above
                    A_pos = -1*pos # Short A
                    A_open = P_a
                    B_pos = 1*pos # Long B
                    B_open = P_b
                    trade = True
                    portfolio_val -= fixed_tc
                    minute_pnl[k] -= fixed_tc
                    print('I am long ', tckr2, ' and am short ', tckr1, ' at ', test_data.loc[(test_data.Sym == tckr2),'Dates'].iloc[k])
                    print('A position ',A_pos)
                    print('B position ',B_pos)
                elif ((error_dist[k] <= -entry) & (error_dist[k-1]> -entry) & (~trade) & (test_data.loc[(test_data.Sym == tckr2),'Dates'].iloc[k].time() < time(15,30))): # Entry from below
                    A_pos = 1*pos #long A
                    A_open = P_a
                    B_pos = -1*pos #Short B
                    B_open = P_b
                    trade = True
                    portfolio_val -= fixed_tc
                    minute_pnl[k] -= fixed_tc
                    print('I am long ', tckr1, ' and am short ', tckr2, ' at ', test_data.loc[(test_data.Sym == tckr2),'Dates'].iloc[k])
                    print('A position ',A_pos)
                    print('B position ',B_pos)
    

        corr_sorted.loc[(tckr1, tckr2),'total_pnl'] = sum(minute_pnl)
        df = pd.DataFrame(error_dist, columns=["Error Distance"])
        df.to_csv(_docsubset + '/' + 'error_dist-freq' + str(frequency)+ '-' + tckr1.replace('/','-') + '-' + tckr2.replace('/','-') + '.csv',index=False)
               
        
        
    portfolio_results.Portfolio_PNL.iloc[i] = sum(corr_sorted.total_pnl)
    corr_sorted.to_csv(_docsubset + '/' + test_date.strftime('%Y-%m-%d-') + 'results-freq' + str(frequency) +'.csv')
    
    #portfolio_val += portfolio_results['Portfolio_PNL'].iloc[i]
    
portfolio_results.to_csv(_doc + '/Portfolio-Returns-freq' + str(frequency) + '.csv')

trade_statistics = pd.DataFrame(columns=['Good Trades','Bad Trades','Hit Ratio','Total PNL','Returns'])
trade_statistics.loc[0,'Good Trades'] = good_trade
trade_statistics.loc[0,'Bad Trades'] = bad_trade
trade_statistics.loc[0,'Hit Ratio'] = good_trade/(good_trade+bad_trade)
trade_statistics.loc[0,'Total PNL'] = portfolio_val - init_portfolio
trade_statistics.loc[0,'Returns'] = (portfolio_val - init_portfolio)/init_portfolio


trade_statistics.to_csv(_doc + '/Trade-Statistics.csv')