import tensorflow as tf
from tensorflow.keras import datasets
from classifiers import knnClassifierFunction
from classifiers import nearestCenterClassifierFunction

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# print(f'the before train images are: {train_images}')
# print(f'the before test images are: {test_images}')


train_images, test_images = train_images / 255.0, test_images / 255.0

# print(f'the after train images are: {train_images}')
# print(f'the after test images are: {test_images}')


# Flatten the images for the classification
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# print(f'the flattened train images are: {train_images_flat}')
# print(f'the flattened test images are: {test_images_flat}')


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


knnClassifierFunction(train_images_flat,test_images_flat, train_labels, test_labels, class_names, 1)
knnClassifierFunction(train_images_flat,test_images_flat, train_labels, test_labels, class_names, 3)
nearestCenterClassifierFunction(train_images_flat, train_labels, test_images_flat, test_labels, class_names)
