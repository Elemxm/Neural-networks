from keras import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC


# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 0 --> Airplane
# 1 --> Automobile
selected_classes = [0, 1]
class_names = ['airplane', 'automobile']

# Use np.where to get the indices of selected classes
train_indices = np.where(np.isin(train_labels, selected_classes))[0]
test_indices = np.where(np.isin(test_labels, selected_classes))[0]

# Filter the dataset based on the selected indices
train_images = train_images[train_indices]
train_labels = train_labels[train_indices]
test_images = test_images[test_indices]
test_labels = test_labels[test_indices]
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

train_labels = train_labels.ravel()
test_labels = test_labels.ravel()

print('SVC Linear Classification results:')
SVC_clf = svm.SVC(kernel='linear')
SVC_clf.fit(train_images, train_labels)
train_images_prediction = SVC_clf.predict(train_images)
training_data_accuracy = accuracy_score(train_labels, train_images_prediction)
print(f'the accuracy score on training data is: {training_data_accuracy}')

test_images_prediction = SVC_clf.predict(test_images)
testing_data_accuracy = accuracy_score(test_labels, test_images_prediction)
print(f'the accuracy score on testing data is: {testing_data_accuracy}')

print('NuSVC Linear Classification results:')
NuSVC_clf = NuSVC(kernel = 'linear', nu=0.5)
NuSVC_clf.fit(train_images, train_labels)
train_images_prediction = NuSVC_clf.predict(train_images)
training_data_accuracy = accuracy_score(train_labels, train_images_prediction)
print(f'the accuracy score on training data is: {training_data_accuracy}')

test_images_prediction = NuSVC_clf.predict(test_images)
testing_data_accuracy = accuracy_score(test_labels, test_images_prediction)
print(f'the accuracy score on testing data is: {testing_data_accuracy}')