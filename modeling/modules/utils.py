# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 02:27:36 2024

@author: Group 13
"""

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np


#%%
"""
Data Preprocessing

First preprocess the data, it involves normalizing the time series data
and creating sequences which we will input to the model.
"""

def preprocess_data(data, sequence_length):
    # Assume 'data' is your sequence of values.
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))
    X, y = create_sequences(data_normalized, sequence_length)
    return X, y
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
def sequencing_data(data):
    data = np.random.rand(100,1)
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_normalized = scaler.fit_transform(data)
    
    # convert to sequences
    def create_sequences(data, sequence_length):
        xs, ys = [],[]
        for i in range(len(data) - sequence_length):
            x = data[i:(i + sequence_length)]
            y = data[i + sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    sequence_length = 10
    X, y = create_sequences(data_normalized, sequence_length)
    return X, y


