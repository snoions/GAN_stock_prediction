#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math

def load_data(path):
    # TODO: test, train split
    x_train = np.array(pd.read_csv(path).drop([0]).drop(columns=['Date']))
    return x_train
            
# data processing functions

def merge_time_series(arr1, arr2) -> tf.float64 :
    t1 = tf.cast(arr1, 'float64') if tf.is_tensor(arr1) else tf.convert_to_tensor(arr1, dtype='float64')
    t2 = tf.cast(arr2, 'float64') if tf.is_tensor(arr2) else tf.convert_to_tensor(arr2, dtype='float64')
    return tf.concat([t1, t2 ], axis=1)

def split_time_series(len_i,len_o, arr) -> (np.array, np.array):
    a = []
    b = []
    for i in range(len(arr)-len_i-len_o + 1):
        a.append(arr[i: i+len_i]) 
        b.append(arr[i+len_i:i+len_i+len_o])
    return (np.array(a), np.array(b))

def moving_average(step_o, arr) -> np.array:
    res = []
    for i in range(len(arr)-step_o):
        a = arr[i: i+step_o]
        res.append(sum(a)/len(a))
    return np.array(res)

def scalers_extract_mean(step_o, scalers) -> np.array:
    res = []
    for s in scalers:
        res.append([s.mean_] * step_o)
    return np.array(res)

def split_time_series_disjoint(len_i,len_o, arr) -> (np.array, np.array):
    a = []
    b = []
    for i in range(0, len(arr)-len_i-len_o + 1,  len_o):
        a.append(arr[i: i+len_i])
        b.append(arr[i+len_i:i+len_i+len_o])
    return (np.array(a), np.array(b))

def batch_standardize(input):
    def standardize_fn(i) : 
        scaler = StandardScaler();
        scaler.fit(i)
        return (scaler.transform(i), scaler)
    return zip(*[ standardize_fn (i) for i in input])

def batch_transform(input, scalers) -> np.array:
    scale = lambda x,s : s.transform(x) if(len(x.shape)==2) else s.transform([x])[0]
    return np.array([scale(x, s) for x,s in zip(input, scalers)])

def batch_inverse_transform(input, scalers) -> np.array:
    scale = lambda x,s : s.inverse_transform(x) if(len(x.shape)==2) else s.inverse_transform([x])[0]
    return np.array([scale(x, s) for x,s in zip(input, scalers)])

def columnify(arr):
    return np.transpose(arr.reshape(-1,arr.shape[-1]))
                                
# accuracy metrics

def MAE(real_o,pred_o):
    res = mean_absolute_error(real_o,pred_o)
    return res

def RMSE(real_o,pred_o):
    res = mean_squared_error(real_o,pred_o)
    return math.sqrt(res)

def MAPE (real_o,pred_o):
    res = mean_absolute_percentage_error(real_o,pred_o)
    return res

def AR(real_o,pred_o):
    res =0
    for i in range(len(real_o)-1):
        if pred_o[i+1] > pred_o[i]:
            res+=real_o[i+1]-real_o[i]
    return res/(len(real_o)-1)

def metrics(real_o,pred_o):
    return {"MAE" :  MAE(real_o,pred_o), "RMSE" : RMSE(real_o,pred_o), "MAPE" : MAPE(real_o,pred_o),  "AR" : AR(real_o,pred_o)}
    

# In[8]:


#tests
# data = oad_data("./data_stock/SP500_average.csv")
# (input, output) = split_time_series(5, 5, data)
# (djt_input, djt_output) = split_time_series_disjoint(5, 5, data)
# print(f'input shape, {input.shape}, output shape, {output.shape}')
# merged = merge_time_series(input, output)
# standardized = batch_standardize(input)
# MA5 = moving_average(data)
# print("input",input)
# print("output", output)
# print("djt_input",djt_input)
# print("djt_input", djt_output)
# print("merged",merged)
# print("standardized", standardized)
# print("MA5", MA5)



