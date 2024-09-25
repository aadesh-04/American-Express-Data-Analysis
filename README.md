# American Express Data Analysis with Neural Networks

## Project Overview

This project leverages neural networks to analyze and classify data from the American Express dataset using Python, TensorFlow, and Keras. The project focuses on applying various data preprocessing techniques like label encoding, one-hot encoding, and feature scaling to prepare the data for training. The model is a sequential neural network that predicts based on the input features.

## Dataset

The **American Express dataset** is provided in this repository (`American-Express.xlsx`). The features include financial and demographic data, and the target variable indicates classification labels.

## Files in Repository

- **learn.ipynb**: A Jupyter notebook that explains the data preprocessing techniques, demonstrated using a sample COVID dataset.
- **project.ipynb**: The main code file that implements the neural network for the American Express dataset.

## Project Structure

- `project.ipynb`: Main code for data analysis and neural network implementation.
- `learn.ipynb`: Learn data preprocessing steps with a sample COVID dataset.
- `American-Express.xlsx`: Dataset used for model training.
  
## Preprocessing Steps

- Label Encoding for categorical variables
- One-Hot Encoding to handle nominal categories
- Standard Scaling to normalize the feature set
- Splitting the data into training and testing sets

## Model Architecture

- Neural Network with 3 hidden layers.
- Activation functions used: ReLU for hidden layers and Sigmoid for output.
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Metrics: Accuracy

## Training

The model is trained with a batch size of 32 and 120 epochs. The performance is evaluated using accuracy and confusion matrix metrics.

## Installation

To run this project locally:

1. Clone the repository.
2. Install required libraries:
    ```bash
    pip install numpy pandas tensorflow scikit-learn
    ```
3. Run the `project.ipynb` notebook to execute the model on the American Express dataset.

## Usage

You can load the dataset, preprocess it, and train the model by following the steps in `project.ipynb`. For a detailed walkthrough on data preprocessing, check out `learn.ipynb`.

## Results

The model provides predictions for the test set and evaluates the performance through accuracy and confusion matrix.

