# Numbers-Summation
This project implements a simple number summation task using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models.

Project Structure

train.py – Trains the model.

dataset_generation.py – Generates synthetic dataset.

model_saving.py – Saves and loads trained models.

requirements.txt – Lists dependencies.

README.md – Project documentation.

Installation

pip install -r requirements.txt

Running the Project

Train the Model

python train.py

You will be prompted to enter the model type (RNN or LSTM).

Load a Saved Model

python model_saving.py <model_filename>

Dataset

The dataset consists of sequences of random integers, and the target output is their sum.

Dependencies

TensorFlow

NumPy

Scikit-learn

