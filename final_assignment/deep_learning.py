import tensorflow as tf
from keras import layers, datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time

# layer_sizes = [10, 64, 128, 256]
layer_sizes = [32, 64]

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Flatten the images for the classification
train_images_flat = train_images.reshape(train_images.shape[0], -1) #The shape is (50000, 3072)
test_images_flat = test_images.reshape(test_images.shape[0], -1) #The shape is (10000, 3072)

#Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

#This network has 2 hidden layers they both are dense layers and have relu activation
for layer1_size in layer_sizes:
    for layer2_size in layer_sizes:
        NAME = "{}-layer1-{}-layer2-{}".format(layer1_size, layer2_size, int(time.time()))
        tensorboard = TensorBoard(log_dir='D:/old-files/Desktop/thmmu/9o/NeurwnikaDiktua/assignment_logs/{}'.format(NAME))
        model = tf.keras.models.Sequential()
        # Flatten layer
        model.add(layers.Flatten(input_shape=(32, 32, 3)))
        model.add(layers.Dense(128, activation='relu'))

        for l in range(layer1_size):
            model.add(layers.Dense(layer1_size, activation= 'relu'))
        
        for l in range(layer2_size):
            model.add(layers.Dense(layer2_size, activation= 'relu'))

        model.add(layers.Dense(128, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
        # Train the model
        model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels), callbacks=[tensorboard])
       
# Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test loss: {test_loss}')
        print(f'Test accuracy: {test_acc}')
