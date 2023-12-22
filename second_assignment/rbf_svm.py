from keras import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import time
import sys

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# 0 --> Airplane
# 1 --> Automobile
selected_classes = [0, 1]
C = {0.001, 0.01, 0.1, 1, 10, 100, 1000}
gamma = {0.001, 0.01, 0.1, 1, 10, 100}

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

def train_and_evaluate_classifier(clf, train_images, train_labels, test_images, test_labels, i, gamma):
    print(f'{clf.__class__.__name__} Linear Classification results for C = {i} and gamma = {gamma}')
    
    # Training
    start_time = time.time()
    clf.fit(train_images, train_labels)
    end_time = time.time()
    print(f'Training time: {end_time - start_time} seconds')

    # Prediction on training images
    start_time = time.time()
    train_images_prediction = clf.predict(train_images)
    end_time = time.time()
    print(f'Prediction time for train images: {end_time - start_time} seconds')
    training_data_ac = accuracy_score(train_labels, train_images_prediction)
    print(f'The accuracy score on training data is: {training_data_ac}')

    # Prediction on test images
    start_time = time.time()
    test_images_prediction = clf.predict(test_images)
    end_time = time.time()
    print(f'Prediction time for test images: {end_time - start_time} seconds')
    testing_data_ac = accuracy_score(test_labels, test_images_prediction)
    print(f'The accuracy score on testing data is: {testing_data_ac}')


# Redirect stdout to a file
output_file = "svm_results.txt"
with open(output_file, 'w') as f:
    sys.stdout = f  # Redirect stdout to the file

    for g in gamma:
        for i in C:
            classifier = SVC(kernel='rbf', C=i, gamma=g)
            train_and_evaluate_classifier(classifier, train_images, train_labels, test_images, test_labels, i, g)

    sys.stdout = sys.__stdout__  # Reset stdout to its original value

# Inform the user that the results are saved
print(f"Results are saved to {output_file}")