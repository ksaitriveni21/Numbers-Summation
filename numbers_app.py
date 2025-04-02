#!/usr/bin/env python
# coding: utf-8

# In[4]:
from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)  # Define Flask app

# Function to safely load a model
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            return f"Error: Model file '{model_path}' not found."

        model = tf.keras.models.load_model(model_path, compile=False)  # Load without custom objects
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        return model
    except Exception as e:
        return f"Model loading error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None

    if request.method == "POST":
        sequence = request.form.get("sequence")
        model_type = request.form.get("model")

        if sequence:
            try:
                numbers = list(map(int, sequence.split(',')))

                if len(numbers) != 3:
                    error = "Please enter exactly three numbers."
                    return render_template("numbers_index.html", error=error, prediction=prediction)

                X_input = np.array(numbers).reshape((1, 3, 1)) / 10.0 

                model_files = {"RNN": "numbers_model_rnn.h5", "LSTM": "numbers_model_lstm.h5"}
                model_path = model_files.get(model_type)

                if not model_path:
                    error = "Invalid model type selected."
                    return render_template("numbers_index.html", error=error, prediction=prediction)

                model = load_model(model_path)

                if isinstance(model, str): 
                    error = model
                else:
                    prediction = model.predict(X_input)[0][0] * 10 

            except Exception as e:
                error = f"Error processing request: {str(e)}"

    return render_template("numbers_index.html", error=error, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  
    app.run(host="0.0.0.0", port=port)


# In[ ]:




