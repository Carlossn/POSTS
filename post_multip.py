# -*- coding: utf-8 -*-

# NEUROMANCER  BLUES: THREADING VS MULTIPROCESSING - PART 2
# https://www.lightbringercap.com/blog/neuromancer-blues-threading-vs-multiprocessing-part-2

# Important Note: don't use IDE such as Sublime or Spyder to run this script
# Multiprocessing modules for windows only allow them to work from a terminal.
# Instructions:
# 1) Select one of the options after line 54
# 2) Comment out the remainder of the lines after line 54
# 3) Open a new terminal
# 4) Type: python file_name.py

import time
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import multiprocessing
import concurrent.futures
from sklearn.svm import SVC
# IO-bound Task - Data download stored in "df" object:
t_list = ['KO','XOM','AAPL','IBM','MCD']
s= '1999-12-31'
e= '2019-12-31' # Last 20 years
df = web.get_data_yahoo(t_list, s,e)
df = df['Adj Close'].copy() # only adjusted closed prices
### CPU-bound Task - SVM model for 5 tickers
def cpu_task(t):
    '''
    Run SVM model using 6 daily return lags as predictors  
    Params
    ------
    t= ticker
    '''
    time.sleep(3)
    lags=6 # number of predictors
    temp=data.filter(regex=t).copy()
    temp[t+'_1d_r']= np.log(temp[t] / temp[t].shift(1))
    for lag in range(1, lags + 1):
        temp[t+'_'+str(lag)+'d_r'] = np.sign(temp[t+'_1d_r'].shift(lag))
    temp[t+'_y'] = np.sign(np.log(temp[t].shift(-1)/temp[t])) # our dependent variable
    temp.dropna(inplace=True)
    X=temp.filter(regex='_r')
    y=temp[t+'_y']
    model = SVC(gamma='auto')
    model.fit(X,y)
    score= model.score(X,y)
    temp[t+'_Pos'] = model.predict(X) # long (1) or Short(-1)
    temp[t+'_strat'] = temp[t+'_Pos'].shift(1) * temp[t+'_1d_r']
    temp[t+'_strat_cum'] = temp[t+'_strat'].cumsum().apply(np.exp)
    stats[t]=[score,temp[t+'_strat_cum'][-1]] # store training score, cum return

#def cpu_task(mock):
#    time.sleep(7)
#    print('done')

##### 1) SEQUENTIAL METHOD: very inefficient
start = time.perf_counter()
data=df.copy()
stats={} # store every ticker results.
for t in data.columns:
    cpu_task(t)
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')

##### 2) MULTIPROCESSING
# A) Manual Multiprocess
start = time.perf_counter()
if __name__=='__main__':
    p1 = multiprocessing.Process(target=cpu_task)
    p2 = multiprocessing.Process(target=cpu_task)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')

# B) Loop Multiprocessinng
start = time.perf_counter()
data=df.copy()
stats={} # store every ticker results.
processes=[]
if __name__=='__main__':
    for t in df.columns: # open a process per ticker
        p = multiprocessing.Process(target=cpu_task, args=[t])
        p.start()
        processes.append(p)    
    for p in processes:
        p.join() # necessary to avoid script to jump to "finish" before processes end
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')

# C) Pool Processing with Multiprocessing module:
start = time.perf_counter()
data=df.copy()
stats={} # stor
if __name__ == '__main__':
    proc_=len(data.columns)
    with multiprocessing.Pool(processes=proc_) as pool:
        pool.map(cpu_task, df.columns)
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')

# D) Pool Processing with concurrent.futures module:
start = time.perf_counter()
data=df.copy()
stats={} # store every ticker results.
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(cpu_task,data.columns)
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')
