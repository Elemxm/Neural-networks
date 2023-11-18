import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def prepare(filepath):
    image_array = cv2.imread(filepath)
    new_array = cv2.resize(image_array, (32, 32))
    flattened_array = new_array.flatten()
    return flattened_array.reshape( -1)


model = tf.keras.models.load_model("first_model")
image_path = 'D:/old-files/Desktop/thmmu/9o/NeurwnikaDiktua/bookie2.jpg'
prediction = model.predict(np.array([prepare(image_path)]))


# print(class_names[int(prediction[0][0])])

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
plt.imshow(image)
plt.title(f'Predicted Class: {class_names[int(prediction[0][0])]}')
plt.show()