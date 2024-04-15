# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 02:27:36 2024

@author: Group 13
"""

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error

#%%
"""
Data Preprocessing

First preprocess the data, it involves normalizing the time series data
and creating sequences which we will input to the model.
"""


# Function to calculate the Simple Moving Average
def SMA(data, window):
    return data.rolling(window=window).mean()

# Function to calculate the Exponential Moving Average
def EMA(data, window):
    return data.ewm(span=window, adjust=False).mean()

# Function to calculate the Relative Strength Index (RSI)
def RSI(data, window):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def preprocess_data(data, features, target, scale_data=True):
    data_scaled = data.copy()
    if scale_data:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Scale all features
        data_scaled[features] = scaler.fit_transform(data[features])
        # Create a separate scaler for the target to be used for inverse transform in predictions
        target_scaler = MinMaxScaler(feature_range=(0, 1)).fit(data[[target]])
    else:
        # If scaling isn't needed, just return the original data and dummy scalers
        scaler = None
        target_scaler = None
    return data_scaled, scaler, target_scaler


def create_sequences(data, n_steps, feature_columns, target_column):
    X, y = [], []
    # Group by 'Company Name' to process each company's data individually
    grouped = data.groupby('Company Name')
    for name, group in grouped:
        # Ensure group is sorted by date, just in case
        group = group.sort_values('Date')
        #print(len(group))
        # Extract features and target from the group
        feature_values = group[feature_columns].values
        target_values = group[target_column].values
        for i in range(n_steps, len(group)):
            X.append(feature_values[i-n_steps:i])  # Sequence of features
            y.append(target_values[i])  # Target value
    return np.array(X), np.array(y)

def calculate_rmse(actual, predicted):
    """
    Calculate the Root Mean Squared Error.
    """
    rmse = sqrt(mean_squared_error(actual, predicted))
    return rmse

def naive_forecast_rmse(actual):
    """
    Benchmark RMSE against a naive forecast (previous time step as prediction).
    """
    naive_forecast = actual[:-1]
    rmse_naive = calculate_rmse(actual[1:], naive_forecast)
    return rmse_naive

def calculate_relative_error(rmse, actual):
    """
    Calculate the relative error of the RMSE with respect to the average actual value.
    """
    average_actual = np.mean(actual)
    relative_error = rmse / average_actual
    return relative_error

def calculate_error_margin(rmse, average_price):
    """
    Calculate the error margin from RMSE as a percentage of the average price.
    """
    error_margin = rmse / average_price * 100
    return error_margin

def volatility_analysis(actual):
    """
    Analyze volatility in terms of standard deviation of actual prices.
    """
    standard_deviation = np.std(actual)
    return standard_deviation