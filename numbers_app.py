#!/usr/bin/env python
# coding: utf-8

# In[4]:

from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)  # ✅ Ensure Flask app is defined before using @app.route

# Function to load model safely
def load_model(model_name):
    try:
        model = tf.keras.models.load_model(model_name, compile=False)  # ✅ Avoid 'mse' issue
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
        return model
    except Exception as e:
        return str(e)

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
                X_input = np.array(numbers).reshape((1, 3, 1)) / 10.0  

                model_path = f"numbers_model_{model_type}.h5"  # Model filename based on user input
                model = load_model(model_path)

                if isinstance(model, str):  # If model loading failed
                    error = f"Model loading error: {model}"
                else:
                    prediction = model.predict(X_input)[0][0] * 10  
            except Exception as e:
                error = f"Error processing request: {str(e)}"
    
    return render_template("numbers_index.html", error=error, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # ✅ Dynamic port for Render
    app.run(host="0.0.0.0", port=port)

# In[ ]:




