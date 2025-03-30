# Numbers-Summation

This project implements a simple number summation task using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models.

## 📂 Project Structure

The repository contains the following files:

- **`model.py`** – Trains the model.
- **`dataset_generation.py`** – Generates a synthetic dataset.
- **`model_saving.py`** – Saves and loads trained models.
- **`requirements.txt`** – Lists dependencies required to run the project.
- **`README.md`** – Project documentation.

## 📥 Installation

Before running the project, install the required dependencies:

```sh
pip install -r requirements.txt
```

## 🚀 Running the Project

### 🏋️ Train the Model

To train the model, run:

```sh
python model.py
```

You will be prompted to enter the model type (`RNN` or `LSTM`).

### 📂 Load a Saved Model

To load a previously saved model, run:

```sh
python model_saving.py <model_filename>
```

Replace `<model_filename>` with the name of your saved model file.

## 📊 Dataset

The dataset consists of sequences of random integers, and the target output is their sum. It is generated synthetically using `dataset_generation.py`.

## 📦 Dependencies

This project requires the following libraries:

- `TensorFlow`
- `NumPy`
- `Scikit-learn`

Ensure that these dependencies are installed before running the scripts.

Feel free to contribute or raise any issues if needed! 🚀

