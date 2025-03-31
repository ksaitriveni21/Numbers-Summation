#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request

app = Flask(__name__)

# Dummy feature names (replace with actual feature names)
feature_names = ["Median Income", "House Age", "Total Rooms", "Total Bedrooms", "Population", "Households", "Latitude", "Longitude"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    classification = None
    error = None

    if request.method == "POST":
        try:
            # Convert input values to float
            values = [float(request.form[feature]) for feature in feature_names]
            # Dummy model prediction (replace with actual ML model)
            prediction = sum(values) * 1000  # Example: Sum * 1000 (for demonstration)
            classification = "Expensive" if prediction > 200000 else "Affordable"
        except ValueError:
            error = "Invalid input. Please enter numeric values."

    return render_template("index.html", feature_names=feature_names, prediction=prediction, classification=classification, error=error)

# Run the Flask app inside Jupyter Notebook
from werkzeug.serving import run_simple

run_simple('localhost', 5000, app)


# In[ ]:




