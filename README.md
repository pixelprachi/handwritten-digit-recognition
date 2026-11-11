# Handwritten Digit Recognition

A CNN trained on the MNIST dataset (99.2% test accuracy) and served as a Flask web app. Draw a digit on the canvas or upload a photo, and the model predicts it with a confidence score.

## Features
- Drawing canvas with live prediction
- Image upload prediction
- Confidence bar and probability for every digit
- Prediction history and session stats

## Tech stack
Python, TensorFlow/Keras, Flask, HTML/CSS/JavaScript

## Run it
pip3 install -r requirements.txt
python3 app.py

Then open http://localhost:5000

The trained model (mnist_model.h5) is included. To retrain it, run python3 train_model.py first.
