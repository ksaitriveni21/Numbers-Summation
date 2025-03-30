from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load Trained Models
rnn_model = tf.keras.models.load_model("numbers_model_rnn.h5")
lstm_model = tf.keras.models.load_model("numbers_model_lstm.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = None

    if request.method == "POST":
        input_seq = request.form.get("sequence")  # Get input from form
        model_type = request.form.get("model")  # Get selected model

        if input_seq:
            # Convert input to numpy array
            num_list = np.array([int(num) for num in input_seq.split(",")]).reshape(1, 3, 1)

            # Predict using the selected model
            if model_type == "RNN":
                prediction = rnn_model.predict(num_list)[0][0]
                selected_model = "Simple RNN"
            else:
                prediction = lstm_model.predict(num_list)[0][0]
                selected_model = "LSTM"

    return render_template("index.html", prediction=prediction, model=selected_model)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
