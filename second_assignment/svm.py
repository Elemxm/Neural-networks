from keras import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 0 --> Airplane
# 1 --> Automobile
selected_classes = [0, 1]

# Use np.where to get the indices of selected classes
train_indices = np.where(np.isin(train_labels, selected_classes))[0]
test_indices = np.where(np.isin(test_labels, selected_classes))[0]

# Filter the dataset based on the selected indices
train_images = train_images[train_indices]
train_labels = train_labels[train_indices]
test_images = test_images[test_indices]
test_labels = test_labels[test_indices]
print(train_images.shape)
print(train_labels.shape)

train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
print(train_images.shape)
print(train_labels.shape)

# classification = svm.SVC()
# classification.fit(train_images, train_labels)
# train_images_prediction = classification.predict(train_images)
# training_data_accuracy = accuracy_score(train_labels, train_images_prediction)
# print(f'the accuracy score on training data is: {training_data_accuracy}')

# test_images_prediction = classification.predict(test_images)
# testing_data_accuracy = accuracy_score(test_labels, test_images_prediction)
# print(f'the accuracy score on testing data is: {testing_data_accuracy}')