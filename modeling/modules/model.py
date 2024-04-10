# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 02:02:46 2024

@author: Group 13
"""
import tensorflow as tf
#%%

"""
Building the LSTM model using Tensorflow
"""

from tensorflow.compat.v1.nn import dynamic_rnn
from tensorflow.compat.v1.keras.layers import LSTMCell, GRUCell


def get_optimizer_loss(sequence_length):
    # Placeholder for input sequences
    sequence_input = tf.compat.v1.placeholder(tf.float32, [None, sequence_length, 1])
    
    # LSTM cell
    lstm_cell = LSTMCell(num_units = 50)
    outputs, states = dynamic_rnn(lstm_cell, sequence_input, dtype=tf.float32)
    
    # Output layer
    logits = tf.compat.v1.layers.dense(states.h, 1)
    
    # Labels placeholder
    labels = tf.compat.v1.placeholder(tf.float32, [None, 1])
    
    # Loss and optimizer
    loss = tf.reduce_mean(tf.square(logits - labels))
    optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)
    
    return loss, optimizer, sequence_input, labels


#%% Custom lstm model

class CustomLSTMModel(tf.Module):
    def __init__(self, units, output_size, sequence_length):
        super(CustomLSTMModel, self).__init__()
        self.sequence_length = sequence_length
        self.units = units
        self.output_size = output_size

        # Initialize weights and biases for LSTM cell
        self.Wxh = tf.Variable(tf.random.normal([self.sequence_length, self.units]), name='Wxh')
        self.Whh = tf.Variable(tf.random.normal([self.units, self.units]), name='Whh')
        self.bias_h = tf.Variable(tf.zeros([self.units]), name='bias_h')

        # Initialize weights and biases for output layer
        self.Who = tf.Variable(tf.random.normal([self.units, self.output_size]), name='Who')
        self.bias_o = tf.Variable(tf.zeros([self.output_size]), name='bias_o')

    def __call__(self, x):
        # Implement the forward pass
        # Note: You'll need to manually implement the LSTM cell operations here
        pass  # Placeholder for LSTM operations

def get_optimizer_loss():
    # Define your custom optimizer and loss function here
    pass  # Placeholder for optimizer and loss function

#%%


def create_get_lstm_model_v1(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
