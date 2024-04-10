# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 03:03:56 2024

@author: Group 13
"""

import tensorflow as tf
import numpy as np
from modules.model import get_optimizer_loss, create_get_lstm_model_v1
from modules.utils import sequencing_data, preprocess_data
#%%

"""
X, y = sequencing_data([])
loss, optimizer, sequence_input, labels = get_optimizer_loss(sequence_length=10)
init = tf.compat.v1.global_variables_initializer()
# Start with tensorflow session
with tf.compat.v1.Session() as sess:
    sess.run(init)
    
    # Example training loop
    epochs = 100
    for epoch in range(epochs):
        _, epoch_loss = sess.run([optimizer, loss], feed_dict={sequence_input: X, labels: y.reshape(-1,1)})
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
"""
# Placeholder for your data loading logic
data = np.random.rand(100)  # Replace with your actual time series data
sequence_length = 10

# Preprocess the data
X, y = preprocess_data(data, sequence_length)

# Create the LSTM model
model = create_get_lstm_model_v1((sequence_length, 1))

# Train the model
model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)            