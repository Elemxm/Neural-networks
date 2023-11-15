from sklearn import model_selection
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Flatten the images for the classification
train_images_flat = train_images.reshape(train_images.shape[0], -1) #The shape is (50000, 3072)
test_images_flat = test_images.reshape(test_images.shape[0], -1) #The shape is (10000, 3072)

#Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Input Layer (3072 neurons)
#   |
# Hidden Layer 1 (128 neurons, ReLU activation)
#   |
# Hidden Layer 2 (64 neurons, ReLU activation)
#   |
# Output Layer (10 neurons, Softmax activation)

# Build the neural network model
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(3072,)))  # Hidden layer with ReLU activation
model.add(layers.Dense(64, activation='relu'))  # Another hidden layer with ReLU activation
model.add(layers.Dense(10, activation='softmax'))  # Output layer with Softmax activation

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

# Train the model
model.fit(train_images_flat, train_labels, epochs=10, validation_data=(test_images_flat, test_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images_flat, test_labels)
print(f'Test accuracy: {test_acc}')
