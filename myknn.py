import tensorflow as tf
from tensorflow.keras import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestCentroid

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Flatten the images for the classification
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)

#Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def knnClassifierFunction(train_images_flat,test_images_flat, train_labels, test_labels, class_names, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(train_images_flat, train_labels.ravel())  # ravel() is used to flatten the labels

    # Make predictions using the test data 
    predictions = knn.predict(test_images_flat)

    # Calculate and print the accuracy 
    accuracy = accuracy_score(test_labels, predictions)
    print(f'Accuracy of KNN Classifier with k = {n_neighbors} : {accuracy * 100:.2f}%')

    # Calculate confusion matrix and classification report 
    confusion_mtx = confusion_matrix(test_labels, predictions)
    print(f'Confusion Matrix for k= {n_neighbors} :')
    print(confusion_mtx)

    class_report = classification_report(test_labels, predictions, target_names=class_names)
    print(f'The classification Report for k = {n_neighbors} :')
    print(class_report)


def nearestCenterClassifierFunction(train_images_flat, train_labels, test_images_flat, test_labels, class_names):
    # Instantiate and train the Nearest Center Classifier
    nearest_center = NearestCentroid()
    nearest_center.fit(train_images_flat, train_labels.ravel())

    # Make predictions using the test data for Nearest Center Classifier
    predictions_nearest_center = nearest_center.predict(test_images_flat)

    # Calculate and print the accuracy for Nearest Center Classifier
    accuracy_nearest_center = accuracy_score(test_labels, predictions_nearest_center)
    print(f'Accuracy of Nearest Center Classifier: {accuracy_nearest_center * 100:.2f}%')

    # Calculate confusion matrix and classification report for Nearest Center Classifier
    confusion_mtx_nearest_center = confusion_matrix(test_labels, predictions_nearest_center)
    print("Confusion Matrix for Nearest Center Classifier:")
    print(confusion_mtx_nearest_center)

    class_report_nearest_center = classification_report(test_labels, predictions_nearest_center, target_names=class_names)
    print("The Classification Report for Nearest Center Classifier:")
    print(class_report_nearest_center)

knnClassifierFunction(train_images_flat,test_images_flat, train_labels, test_labels, class_names, 1)
knnClassifierFunction(train_images_flat,test_images_flat, train_labels, test_labels, class_names, 3)
nearestCenterClassifierFunction(train_images_flat, train_labels, test_images_flat, test_labels, class_names)