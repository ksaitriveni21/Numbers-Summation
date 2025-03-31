#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings("ignore")

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to Generate Dataset
def generate_data(num_samples=5000, seq_length=3):
    X = np.random.randint(1, 10, size=(num_samples, seq_length))
    y = np.sum(X, axis=1).reshape(-1, 1)  # Sum as target output
    return X, y

# Generate Training & Validation Data
X_train, y_train = generate_data()
X_val, y_val = generate_data(num_samples=1000)

# Normalize & Reshape for RNN/LSTM Input
X_train = X_train.reshape((-1, 3, 1)) / 10.0
X_val = X_val.reshape((-1, 3, 1)) / 10.0
y_train, y_val = y_train / 10.0, y_val / 10.0

# Define RNN Model
rnn_model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(3, 1)),
    SimpleRNN(50),
    Dense(25, activation='relu'),
    Dense(1)
])

rnn_model.compile(optimizer=Adam(learning_rate=0.005), loss="mse", metrics=['mae'])

print("Training RNN model...")
history_rnn = rnn_model.fit(X_train, y_train, epochs=20, batch_size=64, 
                            validation_data=(X_val, y_val), verbose=1)

rnn_model.save("numbers_model_rnn.h5")
print("RNN Model saved as 'numbers_model_rnn.h5'")

# Define LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(3, 1)),
    LSTM(50),
    Dense(25, activation='relu'),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.005), loss="mse", metrics=['mae'])

print("Training LSTM model...")
history_lstm = lstm_model.fit(X_train, y_train, epochs=20, batch_size=64, 
                              validation_data=(X_val, y_val), verbose=1)

lstm_model.save("numbers_model_lstm.h5")
print("LSTM Model saved as 'numbers_model_lstm.h5'")

# Save Training Histories
np.save("rnn_history.npy", history_rnn.history)
np.save("lstm_history.npy", history_lstm.history)
print("Training histories saved.")

# Load Trained Models
rnn_model = tf.keras.models.load_model("numbers_model_rnn.h5")
lstm_model = tf.keras.models.load_model("numbers_model_lstm.h5")

def predict(sequence, model_type="RNN"):
    """
    Predicts the sum of a sequence using the specified model.

    Args:
    - sequence (str): A comma-separated string of three numbers.
    - model_type (str): "RNN" or "LSTM" (default: "RNN").

    Returns:
    - float: Predicted sum.
    """
    try:
        # Convert input sequence to numpy array
        numbers = np.array(list(map(int, sequence.split(',')))).reshape(1, 3, 1) / 10.0
        
        # Select model
        model = rnn_model if model_type == "RNN" else lstm_model
        
        if model is None:
            raise ValueError(f"The model for '{model_type}' is not loaded properly.")

        # Make prediction
        prediction = model.predict(numbers, verbose=0)
        
        # Convert back to original scale
        return round(float(prediction[0]) * 10, 2)
    
    except Exception as e:
        return f"Error: {str(e)}"



