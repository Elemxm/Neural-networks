import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestCentroid


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
    print(f'Classification Report for k = {n_neighbors} :')
    print(class_report)

    # # Create a boolean mask for incorrect predictions
    # incorrect_mask = test_labels.ravel() != predictions

    # # Visualize incorrectly classified images for k=3
    # incorrect_images = test_images[incorrect_mask]
    # incorrect_labels = test_labels[incorrect_mask]
    # incorrect_predictions = predictions[incorrect_mask]

    # plt.figure(figsize=(10, 10))
    # for i in range(min(25, len(incorrect_images))):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(incorrect_images[i], cmap=plt.cm.binary)
    #     predicted_label = class_names[incorrect_predictions[i]]
    #     true_label = class_names[incorrect_labels[i][0]]
    #     plt.xlabel(f'Pred: {predicted_label}\nTrue: {true_label}')

    # plt.show()


def nearestCenterClassifierFunction(train_images_flat, train_labels, test_images_flat, test_labels, class_names):
    # Instantiate and train the Nearest Center Classifier
    nearest_center = NearestCentroid()
    nearest_center.fit(train_images_flat, train_labels.ravel())

    # Make predictions using the test data for Nearest Center Classifier with k=1
    predictions_nearest_center = nearest_center.predict(test_images_flat)

    # Calculate and print the accuracy for Nearest Center Classifier with k=1
    accuracy_nearest_center = accuracy_score(test_labels, predictions_nearest_center)
    print(f'Accuracy of Nearest Center Classifier: {accuracy_nearest_center * 100:.2f}%')

    # Calculate confusion matrix and classification report for Nearest Center Classifier
    confusion_mtx_nearest_center = confusion_matrix(test_labels, predictions_nearest_center)
    print("Confusion Matrix for Nearest Center Classifier:")
    print(confusion_mtx_nearest_center)

    class_report_nearest_center = classification_report(test_labels, predictions_nearest_center, target_names=class_names)
    print("Classification Report for Nearest Center Classifier:")
    print(class_report_nearest_center)