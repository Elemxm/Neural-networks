import tensorflow as tf
from keras import layers, models, datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import time

NAME = "cifar10-32x64x64x10-{}".format(int(time.time()))
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

# Input Layer (32x32x3 pixels)
#    |
# Convolutional Layer 1 (32 filters, 3x3 kernel, ReLU activation)
#    |
# Max Pooling Layer 1 (2x2 pool size)
#    |
# Convolutional Layer 2 (64 filters, 3x3 kernel, ReLU activation)
#    |
# Max Pooling Layer 2 (2x2 pool size)
#    |
# Convolutional Layer 3 (64 filters, 3x3 kernel, ReLU activation)
#    |
# Flatten Layer
#    |
# Fully Connected Layer 1 (64 neurons, ReLU activation)
#    |
# Output Layer (10 neurons, Softmax activation)

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation= 'relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images_flat, test_labels)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')


model.save('second_model')
new_model = tf.keras.models.load_model('second_model')
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
sample_images = test_images_flat[indices]
sample_labels = test_labels[indices]

# Make predictions on the selected subset
sample_predictions = new_model.predict(test_images_flat[indices])

# Plot the images with predictions and actual labels
plot_images(sample_images, sample_predictions, sample_labels, class_names)