import tensorflow as tf
from keras import layers, models, datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time

NAME = "256-layer1-256-layer2-{}".format(int(time.time()))
TensorBoard = TensorBoard(log_dir='C:/elenamach/assignment-logs/2e-fixed-dropout/{}'.format(NAME))


# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

#Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# Build the neural network model
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Dropout(0.5))  # Dropout layer after the first Conv2D layer

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout layer after the second Conv2D layer

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_split=0.3, callbacks=[TensorBoard])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

model.save('second_experiment')
new_model = tf.keras.models.load_model('second_experiment')
predictions = new_model.predict([test_images])

# Function to display images along with their predicted and actual labels
def plot_images(images, predicted_labels, actual_labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)

        predicted_label = np.argmax(predicted_labels[i])
        actual_label = actual_labels[i, 0]

        plt.xlabel(f'Predicted: {class_names[predicted_label]} Actual: {class_names[actual_label]}')

    plt.show()

# Select a random subset of test images
num_samples = 25
indices = np.random.choice(len(test_images), num_samples, replace=False)
sample_images = test_images[indices]
sample_labels = test_labels[indices]

# Make predictions on the selected subset
sample_predictions = new_model.predict(test_images[indices])

# Plot the images with predictions and actual labels
plot_images(sample_images, sample_predictions, sample_labels, class_names)