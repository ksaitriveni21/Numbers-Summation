#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, render_template, request
from numbers_model import predict  # Import predict function

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None
    model_type = "RNN"  # Default model type

    if request.method == "POST":
        sequence = request.form.get("sequence")
        model_type = request.form.get("model", "RNN")

        if sequence:
            prediction = predict(sequence, model_type)
            if isinstance(prediction, str) and "Error" in prediction:
                error = prediction

    return render_template("numbers_index.html", error=error, prediction=prediction, model=model_type)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)



# In[ ]:




