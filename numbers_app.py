#!/usr/bin/env python
# coding: utf-8

# In[4]:
#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)  # Ensure Flask app is defined before using @app.route

# Function to load model safely
def load_model(model_name):
    try:
        # Attempt to load the model without compilation to avoid issues with custom objects
        model = tf.keras.models.load_model(model_name, compile=False)
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
                # Parse and reshape the sequence
                numbers = list(map(int, sequence.split(',')))
                if len(numbers) != 3:
                    error = "Please enter exactly three numbers."
                    return render_template("numbers_index.html", error=error, prediction=prediction)

                X_input = np.array(numbers).reshape((1, 3, 1)) / 10.0  # Normalize the input
                
                # Dynamically load the model based on user selection
                if model_type == "RNN":
                    model_path = "numbers_model_rnn.h5"  # Correct model filename for RNN
                elif model_type == "LSTM":
                    model_path = "numbers_model_lstm.h5"  # Correct model filename for LSTM
                else:
                    error = "Invalid model type selected."
                    return render_template("numbers_index.html", error=error, prediction=prediction)

                # Load the selected model
                model = load_model(model_path)

                if isinstance(model, str):  # If model loading failed
                    error = f"Model loading error: {model}"
                else:
                    # Make prediction
                    prediction = model.predict(X_input)[0][0] * 10  # Rescale to original scale
            except Exception as e:
                error = f"Error processing request: {str(e)}"
    
    return render_template("numbers_index.html", error=error, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Dynamic port for Render
    app.run(host="0.0.0.0", port=port)



# In[ ]:




