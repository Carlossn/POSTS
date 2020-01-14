# Modules
import time
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

# data download:
t_list = ['KO']
s= '1999-12-31'
e= '2018-12-31' # Training Period 2000-2018
df = web.get_data_yahoo(t_list, s,e) 
df = df['Adj Close'].copy() # Only Price

# feature engineering: creating predictors
t=df.columns # ticker
# SMA Rule Signals:   
SMA_L=200
SMA_M=50
SMA_S=10
df['SMA_L'] = df[t].rolling(SMA_L).mean()  
df['SMA_M'] = df[t].rolling(SMA_M).mean()  
df['SMA_S'] = df[t].rolling(SMA_S).mean()  
df['X_ML_r']= np.where(df['SMA_M'] > df['SMA_L'], 1, -1)  
df['X_SM_r']= np.where(df['SMA_S'] > df['SMA_M'], 1, -1) 
df['X_SL_r']= np.where(df['SMA_S'] > df['SMA_L'], 1, -1)  

# Lag returns binary signals:
lags=6 # number of predictors
df[t+'_1d_r']= np.log(df[t] / df[t].shift(1))
for lag in range(1, lags + 1):
    df[t+'_'+str(lag)+'d_r'] =df[t+'_1d_r'].shift(lag)

# Define Model X and y
df[t+'_y'] = np.sign(np.log(df[t].shift(-1)/df[t])) # dependent variable = 1 day future return on a binary basis
df.dropna(inplace=True)
X=df.filter(regex='_r').copy()
y=df[t+'_y']
y.head(5)

# train/validation split:
tscv = TimeSeriesSplit(n_splits=2) # generate train/cv indices => this generate 2 sets of train/cv indices
train_idx = list(tscv.split(df))[1][0] # take the second set of train indices
X=X.iloc[train_idx]
y=y.iloc[train_idx]

# Model Training: Train simple Logit Model
model= LogisticRegression()
model.fit(X,y)
model.score(X,y)

#### Approach 1:Pickle
# import library
import pickle

pkl_file = "LOG_model.pkl"  

# Save model in current folder
with open(pkl_file, 'wb') as file:  
    pickle.dump(model, file)
    
# Load Model
with open(pkl_file, 'rb') as file:  
    load_model = pickle.load(file)
# check model:
load_model.score(X,y)
    
#### Approach 2:Joblib
# import library
from sklearn.externals import joblib
# Save model in current folder:
jl_file = "jl_LOG_model.pkl"  
joblib.dump(model, jl_file)

# Load Model
jl_load_model = joblib.load(jl_file)
# Check Model
jl_load_model.score(X,y)


#### Approach 3: Proprietary

## Import libraries
import json

# Reboot model object (Parent Class):
model= LogisticRegression()

# Create Synthetic ML Model Class:
# build model class based on our model instance class e.g. LogisticRegression, SVM, etc
# Note; 
class my_model(model.__class__):    
    def __init__(self):
        model.__class__.__init__(self) # 1

    def mimic(self, model_inst):
        '''
        Create model based on copying an instance of an already created model 
        Parameter
        ---------
        model_inst = enter model object instance with trained/fitted model
        '''
        list_ = list(filter(lambda x: x.endswith('_')!=0 and x.startswith('_')==0 ,dir(model_inst))) # instance attributes (ending "_")
        for i in list_: # 2
            setattr(self,i,eval('model_inst.'+eval('i')))
        
    def save(self,path):
        '''
        Save model information using json format.
        Params
        ------
        path = file name using "name.json" format including path address where it will be stored in a different \
        directory than the current folder.
        
        '''
        dict_={'path':path}
        array_param=[]
        for i in self.__dict__.keys():
            dict_[i]= eval('self.'+eval('i'))
            if type(dict_[i]).__name__== 'ndarray': # 3
                array_param.append(i)
                dict_[i]=dict_[i].tolist() # 4 
        dict_['array_param']=array_param # 5
        json_ = json.dumps(dict_, indent=4)
        with open(path, 'w') as file:
            file.write(json_)
        
    def load(self,path):
        '''
        Load model information using json format.
        Params
        ------
        path = file name using "name.json" format where the model info will loaded \
        including path address if the json file is in a folder different than the \
        current working directory
        '''
        with open(path, 'r') as file:
            dict_load = json.load(file)
            for p in dict_load['array_param']:
                dict_load[p] = np.array(dict_load[p]) # 6
        
        for i in dict_load.keys():
            setattr(self,i,dict_load[i])


# Create new model instance:
test=my_model()
# Create a synthetic replica of our trained ML model object instance:
test.mimic(model)
# Save our new synthetic model configuration:
test.save('test.json')

# Restart your kernel/session
# Load your X,y data again
# Load ML Model:
load_model = my_model()
load_model.load('js.json')
load_model.__dict__ # check trained ML model info (coefficients, etc) is loaded
load_model.score(X,y) # check score