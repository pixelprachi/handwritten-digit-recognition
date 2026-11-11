import tensorflow as tf
from tensorflow import keras
import numpy as np
 
print('Loading MNIST dataset...')
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
 
# Normalize pixel values from 0-255 to 0.0-1.0
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0
 
# Reshape: add channel dimension (28,28) -> (28,28,1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)
 
print('Building the neural network...')
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
 
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2,2),
 
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
 
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
 
    keras.layers.Dense(10, activation='softmax')
])
 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
 
model.summary()
 
print('\nTraining the model — this takes 2 to 5 minutes on M5...')
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)
 
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest Accuracy: {test_accuracy*100:.2f}%')
 
model.save('mnist_model.h5')
print('\nModel saved as mnist_model.h5')
print('Training complete! You can now run app.py')
