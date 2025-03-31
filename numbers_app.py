#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, render_template, request
import os

app = Flask(__name__)

def predict_sum(sequence, model):
    try:
        numbers = list(map(int, sequence.split(',')))
        return sum(numbers)  
    except ValueError:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    model = None
    
    if request.method == "POST":
        sequence = request.form.get("sequence")
        model = request.form.get("model")

        if sequence:
            prediction = predict_sum(sequence, model)
            if prediction is None:
                error = "Invalid input! Please enter three numbers separated by commas."

    return render_template("numbers_index.html", error=error, prediction=prediction, model=model)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


# In[ ]:




