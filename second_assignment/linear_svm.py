from keras import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import NuSVC
import matplotlib.pyplot as plt
import time

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

#Reshaping the imags and labels
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
train_labels = train_labels.ravel()
test_labels = test_labels.ravel()

def plot_images(classifier, test_images, test_labels, class_names):
    num_displayed = 15

    for i in range(min(num_displayed, len(test_images))):
        true_label = test_labels[i]
        
        # Reshape the test image to 2D array
        test_image_2d = test_images[i].reshape(1, -1)

        # Predict using the classifier
        predicted_label = classifier.predict(test_image_2d)


        # Display the image using matplotlib
        plt.subplot(5, 3, i + 1)
        plt.imshow(test_images[i].reshape(32, 32, 3))  # Assuming images are 32x32x3
        plt.title(f'True: {class_names[true_label]} Predicted: {class_names[predicted_label[0]]}')
        plt.axis('off')

    plt.show()

def train_and_evaluate_classifier(clf, train_images, train_labels, test_images, test_labels, class_names):
    print(f'{clf.__class__.__name__} Linear Classification results:')
    
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

    # Plotting images
    plot_images(clf, test_images, test_labels, class_names)

# Train and evaluate SVC
SVC_clf = svm.SVC(kernel='linear')
train_and_evaluate_classifier(SVC_clf, train_images, train_labels, test_images, test_labels, class_names)

# Train and evaluate NuSVC
NuSVC_clf = NuSVC(kernel='linear', nu=0.5)
train_and_evaluate_classifier(NuSVC_clf, train_images, train_labels, test_images, test_labels, class_names)