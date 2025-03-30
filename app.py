from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your trained model
model = load_model("numbers_model.h5")

def predict_sum(sequence):
    sequence = np.array(sequence).reshape(1, -1, 1)  # Reshape for LSTM
    prediction = model.predict(sequence)
    return round(prediction[0][0], 2)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        try:
            numbers = request.form["numbers"]
            sequence = [int(num) for num in numbers.split(",")]
            result = predict_sum(sequence)
        except:
            result = "Invalid input. Enter numbers separated by commas."
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
