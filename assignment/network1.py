import tensorflow as tf
from keras import layers, models, datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time

NAME = "cifar10-3072x128x64x10-{}".format(int(time.time()))
TensorBoard = TensorBoard(log_dir= 'D:/old-files/Desktop/thmmu/9o/NeurwnikaDiktua/logs/{}'.format(NAME))


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
model = tf.keras.models.Sequential()
model.add(layers.Dense(128, activation= tf.nn.relu, input_shape=(3072,)))  # Hidden layer with ReLU activation
model.add(layers.Dense(64, activation= tf.nn.relu))  # Another hidden layer with ReLU activation
model.add(layers.Dense(10, activation= tf.nn.softmax))  # Output layer with Softmax activation

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
              metrics=['accuracy'])

# Train the model
model.fit(train_images_flat, train_labels, epochs=10, validation_data=(test_images_flat, test_labels), callbacks=[TensorBoard])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images_flat, test_labels)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')

model.save('first_model')
new_model = tf.keras.models.load_model('first_model')
predictions = new_model.predict([test_images_flat])

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
indices = np.random.choice(len(test_images_flat), num_samples, replace=False)
sample_images = test_images[indices]
sample_labels = test_labels[indices]

# Make predictions on the selected subset
sample_predictions = new_model.predict(test_images_flat[indices])

# Plot the images with predictions and actual labels
plot_images(sample_images, sample_predictions, sample_labels, class_names)


# plt.imshow(train_images[0],cmap=plt.cm.gray)
# actual_label = train_labels[0, 0]
# plt.xlabel(f'Actual: {class_names[actual_label]}')
# plt.show()