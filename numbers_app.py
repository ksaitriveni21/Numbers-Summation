#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf

def load_model(model_name):
    return tf.keras.models.load_model(model_name)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    prediction = None

    if request.method == "POST":
        sequence = request.form.get("sequence")
        model_type = request.form.get("model")

        if sequence:
            numbers = list(map(int, sequence.split(',')))
            X_input = np.array(numbers).reshape((1, 3, 1)) / 10.0  

            try:
                model = load_model(f"numbers_model_{model_type}.h5")  
                prediction = model.predict(X_input)[0][0] * 10  
            except Exception as e:
                error = f"Error loading model: {str(e)}"
    
    return render_template("numbers_index.html", error=error, prediction=prediction)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))  # Render uses a dynamic port
    app.run(host="0.0.0.0", port=port)


# In[ ]:




