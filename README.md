# Handwritten Digit Classification using Neural Networks (MNIST Dataset)

This project builds a simple neural network to recognize handwritten digits (0–9) using the MNIST dataset.

---

## Project Overview

- **Dataset:** MNIST (70,000 images of 28x28 pixel handwritten digits)
- **Model:** Feedforward neural network with two hidden layers
- **Tools:** Python, TensorFlow, Keras, Google Colab
- **Accuracy:** Around 98% on test data

---

## Approach Summary

1. Loaded the MNIST dataset from TensorFlow.
2. Normalized the pixel values to the 0-1 range.
3. Converted digit labels to one-hot encoded format.
4. Built and trained a simple neural network for 5 epochs.
5. Evaluated the model using accuracy and confusion matrix.
6. Saved the trained model file.

---

## How to Run

1. Open the `Digit_Classification_Project.ipynb` notebook in Google Colab.
2. Run all cells step-by-step.
3. Check the accuracy and sample predictions in the output.

---

## Requirements

```bash
tensorflow
numpy
matplotlib
scikit-learn
