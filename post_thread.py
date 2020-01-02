# -*- coding: utf-8 -*-

# NEUROMANCER  BLUES: THREADING VS MULTIPROCESSING - PART 1
# https://www.lightbringercap.com/blog/neuromancer-blues-threading-vs-multiprocessing-part-1
import numpy as np
import pandas as pd
import concurrent.futures
import threading
import time

### PART 1 -  THREADING
import time
import pandas_datareader.data as web
import pandas as pd
import threading
import concurrent.futures

# Define task/worker function:
def io_task(ticker_, dict_):
    '''
    Download Yahoo Data and store it into dictionary
    '''
    df = web.get_data_yahoo(ticker_)    
    time.sleep(1)
    dict_[ticker_]=df
    return df
# Case a) Sequential
t_list = ['KO','XOM','AAPL','IBM','MCD']
df_dict={}
start = time.perf_counter()
for i in t_list:
    df_dict[i]=io_task(i, df_dict)
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')
print(f'dictionary keys are {df_dict.keys()}')

# Case b) Classic Threading
t_list = ['KO','XOM','AAPL','IBM','MCD']
df_dict={}
threads=[]
start = time.perf_counter()
for i in t_list:
     t = threading.Thread(target=io_task, args=[i,df_dict])
     t.start()
     threads.append(t)
for thread in threads:
     thread.join()
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')
print(f'dictionary keys are {df_dict.keys()}')
df_dict['XOM']

# Case c) Pooling Threading using concurrent.futures
# define helper for execute.map
def helper(inputs):
    io_task(inputs[0], inputs[1])

t_list = ['KO','XOM','AAPL','IBM','MCD']
df_dict={}
args = list(zip(t_list,[df_dict for _ in range(len(t_list))]))

start = time.perf_counter()    
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(helper, args)
finish = time.perf_counter()
print(f'running time: {finish-start} second(s)')
print(f'dictionary keys are {df_dict.keys()}')
