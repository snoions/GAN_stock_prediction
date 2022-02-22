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

#TODO: support multiple output timesteps
def split_time_series(len_i,len_o, arr) -> (np.array, np.array):
    a = []
    b = []
    for i in range(len(arr)-len_i-len_o):
        a.append(arr[i: i+len_i])
        b.append(arr[i+len_i:i+len_i+len_o])
    return (np.array(a), np.array(b))

def split_time_series(len_i,len_o, arr) -> (np.array, np.array):
    a = []
    b = []
    for i in range(len(arr)-len_i-len_o):
        a.append(arr[i: i+len_i])
        b.append(arr[i+len_i:i+len_i+len_o])
    return (np.array(a), np.array(b))

def split_time_series_disjoint(len_i,len_o, arr) -> (np.array, np.array):
    a = []
    b = []
    for i in range(0, len(arr)-len_i-len_o,  len_o):
        a.append(arr[i: i+len_i])
        b.append(arr[i+len_i:i+len_i+len_o])
    return (np.array(a), np.array(b))

def batch_standardize(input):
    def standardize_fn(i) : 
        scaler = StandardScaler();
        scaler.fit(i)
        return (scaler.transform(i), scaler)
    return zip(*[ standardize_fn (i) for i in input])

def batch_transform(input, scalers):
    scale = lambda x,s : s.transform(x) if(len(x.shape)==2) else s.transform([x])[0]
    return np.array(list(map(scale, input, scalers)))

def batch_inverse_transform(input, scalers):
    scale = lambda x,s : s.inverse_transform(x) if(len(x.shape)==2) else s.inverse_transform([x])[0]
    return np.array(list(map(scale, input, scalers)))

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

# In[8]:


#tests
# (input, output) = split_time_series(5, 5, load_data("./data_stock/SP500_average.csv"))
# (djt_input, djt_output) = split_time_series_disjoint(5, 5, load_data("./data_stock/SP500_average.csv"))
# print(f'input shape, {input.shape}, output shape, {output.shape}')
# merged = merge_time_series(input, output)
# standardized = batch_standardize(input)
# print("input",input)
# print("output", output)
# print("djt_input",djt_input)
# print("djt_input", djt_output)
# print("merged",merged)
# print("flattened input", flatten(input))
# print("standardized", standardized)



