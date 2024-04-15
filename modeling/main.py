# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 03:03:56 2024

@author: Group 13
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from modules.model import custom_model, model_train, model_evaluate, model_predict
from modules.utils import create_sequences, preprocess_data, SMA, EMA, RSI, calculate_rmse, naive_forecast_rmse, calculate_relative_error, calculate_error_margin, volatility_analysis
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#%% reading data
finance_laborstats_datapath = './data/finance_laborstats.csv'
finance_laborstats = pd.read_csv(finance_laborstats_datapath)
finance_laborstats['Date'] = pd.to_datetime(finance_laborstats['Date'])
finance_laborstats.set_index('Date', inplace=True)
# Add Technical Indicators as new columns in the DataFrame
finance_laborstats['SMA_20'] = SMA(finance_laborstats['Adj Close'], window=20)  # 20-day simple moving average
finance_laborstats['EMA_20'] = EMA(finance_laborstats['Adj Close'], window=20)  # 20-day exponential moving average
finance_laborstats['RSI_14'] = RSI(finance_laborstats['Adj Close'], window=14)  # 14-day Relative Strength Index
finance_laborstats = finance_laborstats.dropna()

#excluded_companies = ['Albemarle Corporation (NYSE:ALB)', 'Biogen Inc. (NasdaqGS:BIIB)', 'Commerce Bancshares, Inc. (NasdaqGS:CBSH)']
#finance_laborstats_filtered = finance_laborstats[~finance_laborstats['Company Name'].isin(excluded_companies)]

# Fill NaNs with the mean or median of the column
mean_value = finance_laborstats['average_hourly_earnings'].mean()
finance_laborstats['average_hourly_earnings'].fillna(mean_value, inplace=True)

#%% Preprocess Data (i.e., Data Normalization)

features = [
    'SMA_20', 'EMA_20', 'RSI_14', 'Volume', 'Environment Score', 'Social Score', 
    'Governance Score', 'average_hourly_earnings', 
    'Interpolated Employment Cost Index', 'payroll_employment',
    'PPIACO', 'Interpolated Labor Productivity', 'unemployment_rate',
    'Interpolated HDI'
]

target_column = 'Adj Close'  # Target variable for prediction

# finance_laborstats_PreProcess, scaler,target_scaler  = preprocess_data(finance_laborstats, features, target_column)
# Call preprocess_data with scale_data=False since the data is already scaled
finance_laborstats_PreProcess, scaler, target_scaler = preprocess_data(
    finance_laborstats, features, target_column, scale_data=False
)

print("NaNs after preprocessing:", finance_laborstats_PreProcess.isnull().any())
#%% Create Train and Test data respecting company boundaries
# Note: You need to decide how to split your data into train and test sets. A simple way could be to use a date cutoff.
cutoff_date = pd.Timestamp('2020-01-01')  # For example, using January 1, 2020 as cutoff date
train_data = finance_laborstats_PreProcess[finance_laborstats_PreProcess.index < cutoff_date]
test_data = finance_laborstats_PreProcess[finance_laborstats_PreProcess.index >= cutoff_date]

# 21 years * 12 months = 252 data points per company.
# Number of time steps to use for sequence.
n_steps = 20
# X, y = create_sequences(finance_laborstats, n_steps, features, target_column)


#%% Create Train and Test data
# Create sequences
X_train, y_train = create_sequences(train_data, n_steps, features, target_column)
X_test, y_test = create_sequences(test_data, n_steps, features, target_column)

sequence_length = n_steps
num_features = len(features)


X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))



#%% Run model training, evaluation, and prediction

# Define and train the LSTM model
model = custom_model((sequence_length, num_features))
model, history = model_train(model, X_train, y_train, epochs=20, batch_size=16, val_split=0.2)

# Predict and evaluate the model
predicted_stock_price = model_predict(model, X_test, target_scaler)
test_rmse = model_evaluate(y_test, predicted_stock_price) 

#%%
import matplotlib.pyplot as plt

# List of excluded companies
excluded_companies = ['Albemarle Corporation (NYSE:ALB)', 'Biogen Inc. (NasdaqGS:BIIB)', 'Commerce Bancshares, Inc. (NasdaqGS:CBSH)']

# Loop through each excluded company and predict the stock prices
for company_name in excluded_companies:
    company_name_s = company_name.split('(')[0].strip()
    # Select data for a specific company
    company_data = finance_laborstats[finance_laborstats['Company Name'] == company_name]
    
    # Preprocess the entire company data
    company_data_preprocessed, _, target_scaler = preprocess_data(company_data, features, target_column, scale_data=False)
    
    # Create sequences from the entire company data
    X_company, y_company = create_sequences(company_data_preprocessed, n_steps, features, target_column)
    X_company = X_company.reshape((X_company.shape[0], n_steps, len(features)))
    
    # Predict using the model
    predicted_prices = model_predict(model, X_company, target_scaler)
    
    # Flatten predicted_prices if needed
    predicted_prices = predicted_prices.flatten()

    # Actual prices for the entire company data
    actual_prices = company_data_preprocessed[target_column]

    # Ensure the number of predictions matches the number of available dates in company_data
    min_length = min(len(predicted_prices), len(company_data_preprocessed))
    predicted_dates = company_data_preprocessed.index[:min_length]

    # Plot the actual vs. predicted prices for the entire dataset
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_dates, actual_prices.iloc[:min_length], label='Actual Prices', color='blue')
    plt.plot(predicted_dates, predicted_prices[:min_length], label='Predicted Prices', linestyle='--', color='red')
    plt.title(f'Stock Price Prediction for {company_name}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(f'./images/Stock Price Prediction for {company_name_s}.png', transparent=True, bbox_inches='tight', format='png')
    plt.show()



#%% Plot training and validation_loss

# Extracting the training and validation loss
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'./images/TrainingandValidationLoss.png', transparent=True, bbox_inches='tight', format='png')
plt.show()

#%%

# To use these functions:
actual_prices = y_test  # Replace with actual stock prices
predicted_prices = predicted_stock_price  # Replace with your model's predicted stock prices

# Calculate RMSE for the model
model_rmse = test_rmse

# Benchmark with Naive Forecast
naive_rmse = naive_forecast_rmse(actual_prices)

# Relative Error
relative_error = calculate_relative_error(model_rmse, actual_prices)

# Error Margin
average_price = np.mean(actual_prices)
error_margin = calculate_error_margin(model_rmse, average_price)

# Volatility
market_volatility = volatility_analysis(actual_prices)

# Print results
print(f"Model RMSE: {model_rmse}")
print(f"Naive Forecast RMSE: {naive_rmse}")
print(f"Relative Error: {relative_error}")
print(f"Error Margin: {error_margin}%")
print(f"Market Volatility (Std Deviation of Prices): {market_volatility}")