import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math

def load_data(path) -> np.array:
    # TODO: test, train split
    dataset = pd.read_csv(path)
    titles = dataset.head()
    cleaned_data = dataset.drop([0]).drop(columns=['Date'])
    return cleaned_data, titles
            
# data processing functions
def k_fold_index_split(length, num_fold):
    idx = [ i for i in range(0, length)]
    np.random.shuffle(idx)
    batch_size = math.floor(length / num_fold)
    remainder = length % num_fold
    batches = []
    for i in range(0, num_fold):
        start = i * batch_size
        end = start + batch_size + 1 if i < remainder else start + batch_size
        batches.append( idx[start : end])
    return batches
    
def merge_time_series(arr1, arr2) -> tf.float64 :
    t1 = tf.cast(arr1, 'float64') if tf.is_tensor(arr1) else tf.convert_to_tensor(arr1, dtype='float64')
    t2 = tf.cast(arr2, 'float64') if tf.is_tensor(arr2) else tf.convert_to_tensor(arr2, dtype='float64')
    return tf.concat([t1, t2 ], axis=1)

def group_time_series(length, arr) -> np.array:
    res = []
    for i in range(len(arr)-length):
        res.append(arr[i: i+length])
    return np.array(res)
                   
def group_time_series_disjoint(length, arr) -> np.array:
    res = []
    for i in range(0, len(arr)-length, length):
        res.append(arr[i: i+length])
    return np.array(res)

def split_input_output(step_i, step_o, data) -> (np.array, np.array):
    if data.shape[1] != step_i + step_o:
        raise ValueError(f"unable to split data of {data.shape[1]} steps into input of {step_i} steps and output of {step_o} steps")
    input_data = []
    output_data = []   
    for d in data:
      input_data.append(d[:step_i])
      output_data.append(d[step_i : step_i + step_o])
    return (np.array(input_data), np.array(output_data))

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