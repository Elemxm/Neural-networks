import tensorflow as tf
from keras import layers, datasets
from keras.callbacks import TensorBoard
import time

layer_sizes = [10, 64, 128, 256]

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

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
        model.add(layers.Conv2D(layer1_size, (3, 3), activation= 'relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(layer2_size, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        
        # Train the model
        model.fit(train_images, train_labels, epochs=10, validation_split=0.3, callbacks=[tensorboard])
       
        # Evaluate the model on the test set
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print(f'Test loss: {test_loss}')
        print(f'Test accuracy: {test_acc}')
