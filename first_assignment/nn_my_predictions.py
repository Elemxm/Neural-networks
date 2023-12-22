import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

models = ['first_experiment', 'second_experiment']

def prepare(filepath):
    image_array = cv2.imread(filepath)
    new_array = cv2.resize(image_array, (32, 32))
    # Ensure the array has 3 channels (assuming it's a color image)
    if new_array.shape[-1] != 3:
        new_array = cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB)
    return new_array.reshape(-1, 32, 32, 3)


# Path to the image I want to predict
image_path = 'D:/old-files/Desktop/thmmu/9o/NeurwnikaDiktua/ship.jpg'

# Preparing the image for prediction
input_image = prepare(image_path)

for model in models: 
    # Loading the pre-trained model
    model = tf.keras.models.load_model(model)
    # Making predictions
    prediction = model.predict(input_image)

    # Displaying the original image and predicted class
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(image)
    plt.title(f'Predicted Class: {class_names[np.argmax(prediction)]}')
    plt.show()