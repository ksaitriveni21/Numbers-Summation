#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

def generate_data(num_samples=5000, seq_length=3):
    X, y = [], []
    for _ in range(num_samples):
        seq = np.random.randint(1, 10, size=(seq_length,))  
        X.append(seq)
        y.append([np.sum(seq)])  
    return np.array(X), np.array(y)

X_train, y_train = generate_data()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

rnn_model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(3, 1)),
    SimpleRNN(50),
    Dense(25, activation='relu'),
    Dense(1)  
])

rnn_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])

print("Training RNN model...")
rnn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

rnn_model.save("numbers_model_rnn.h5")
print("RNN Model saved as 'numbers_model_rnn.h5'")

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(3, 1)),
    LSTM(50),
    Dense(25, activation='relu'),
    Dense(1)  
])

lstm_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse', metrics=['mae'])
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
lstm_model.save("numbers_model_lstm.h5")
print("LSTM Model saved as 'numbers_model_lstm.h5'")


# In[ ]:




