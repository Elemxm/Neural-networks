import tensorflow as tf
from keras import layers, datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time


dense_layers = [0, 1, 2]
layer_sizes = [10, 64, 128]
conv_layers = [1, 2, 3]


# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Flatten the images for the classification
train_images_flat = train_images.reshape(train_images.shape[0], -1) #The shape is (50000, 3072)
test_images_flat = test_images.reshape(test_images.shape[0], -1) #The shape is (10000, 3072)

#Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorBoard = TensorBoard(log_dir= 'D:/old-files/Desktop/thmmu/9o/NeurwnikaDiktua/assignment_logs/{}'.format(NAME))
            model = tf.keras.models.Sequential()
            model.add(layers.Conv2D(layer_size, (3, 3), activation= 'relu', input_shape=(32, 32, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            
            for l in range(conv_layer-1):
                model.add(layers.Conv2D(layer_size, (3, 3), activation='relu'))
                model.add(layers.MaxPooling2D((2, 2)))
            
            model.add(layers.Flatten())
            
            for l in range(dense_layer):
                model.add(layers.Dense(layer_size, activation='relu'))
                
                
            model.add(layers.Dense(layer_size, activation='softmax'))

            # Compile the model
            model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
            
            # Train the model
            model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

            # Evaluate the model on the test set
            test_loss, test_acc = model.evaluate(test_images, test_labels)
            print(f'Test loss: {test_loss}')
            print(f'Test accuracy: {test_acc}')