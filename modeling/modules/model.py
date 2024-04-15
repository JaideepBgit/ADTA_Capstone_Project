# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 02:02:46 2024
@author: Group 13
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
#%%
# Global scaler for inverse transforming predictions
scaler = MinMaxScaler(feature_range=(0, 1))

def custom_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)  # Lower learning rate
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def model_train(model, X_train, y_train, epochs, batch_size, val_split):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split, use_multiprocessing=True)
    return model, history

def model_evaluate(y_test, predicted_stock_price):
    rmse = sqrt(mean_squared_error(y_test, predicted_stock_price))
    print(f'Test RMSE: {rmse}')
    return rmse

# Assuming scaler is fit to the training data
def model_predict(model, X_test, target_scaler=None):
    predicted_stock_price = model.predict(X_test)
    if target_scaler:
        predicted_stock_price = target_scaler.inverse_transform(predicted_stock_price)
    return predicted_stock_price
