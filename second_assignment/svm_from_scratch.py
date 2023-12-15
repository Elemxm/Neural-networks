from keras import datasets
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# Loading the CIFAR-10 dataset
# X_train = train_images
# X_test = test_images
# Y_train = train_labels
# Y_testt = test_labels

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
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)



class SVM_classifier():

    #Initiating the hyperparameters
    def __init__(self, learning_rate, iterations, lambda_parameter):
    # y = wx - b
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_parameter = lambda_parameter

    #Fitting the dataset to SVM Classifier   
    def fit(self, X, Y ):
        #rows --> number of Data points ---> number of Rows
        #columns --> number of input features ---> number of Columns
        self.rows, self.columns = X.shape

        #Initiating the weight value and bias value
        self.w = np.zeros(self.columns)
        self.b = 0
        self.X = X
        self.Y = Y

        #Implementing Gradient Descent algorithm for Optimization
        for i in range(self.iterations):
            self.update_weights()
    
    #Updating the weight and bias value
    def update_weights(self):

        #Label encoding
        y_label = np.where(self.Y <= 0, -1, 1)

        # greadients (dw, db)
        for index, x_i in enumerate(self.X):

            condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1

            if(condition == True):
                # dj/dw = 2lw
                # dj/db = 0
                dw = 2 * self.lambda_parameter * self.w
                db = 0

            else:
                # dj/dw = 2lw -yi*xi
                # dj/db = yi
                dw = 2 * self.lambda_parameter * self.w - y_label[index] * x_i
                db = y_label[index]

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

    #Predicting the label for given input value
    def predict(self, X):

        #y = wx - b
        output = np.dot(X, self.w) - self.b
        predicted_labels = np.sign(output)
        
        y_hat = np.where(predicted_labels <= -1, 0, 1)

        return y_hat

#Calling the instance of the Model
classifier = SVM_classifier(learning_rate=0.001, iterations=1000, lambda_parameter=0.01)

#training the SVM classifier with training data

classifier.fit(train_images, train_labels)
train_images_prediction = classifier.predict(train_images)
training_data_accuracy = accuracy_score(train_labels, train_images_prediction)
print(f'the accuracy score on training data is: {training_data_accuracy}')


classifier.fit(test_images, test_labels)
test_images_prediction = classifier.predict(test_images)
testing_data_accuracy = accuracy_score(test_labels, test_images_prediction)
print(f'the accuracy score on testing data is: {testing_data_accuracy}')

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# Perform PCA to reduce the dimensionality to 2
pca = PCA(n_components=2)
train_images_pca = pca.fit_transform(train_images)

# Calling the instance of the Model
classifier = SVM_classifier(learning_rate=0.001, iterations=1000, lambda_parameter=0.01)

# Training the SVM classifier with PCA-transformed training data
classifier.fit(train_images_pca, train_labels)

# Plotting the PCA-transformed data
plt.scatter(train_images_pca[:, 0], train_images_pca[:, 1], c=train_labels, cmap=plt.cm.Paired, marker='o', edgecolors='k')

# Plotting the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Highlight the support vectors
plt.scatter(classifier.X[:, 0], classifier.X[:, 1], s=100, facecolors='none', edgecolors='k', marker='o')

# Plotting predictions with different colors for correct and incorrect
predictions = classifier.predict(train_images_pca)
correct_predictions = (predictions == train_labels.flatten())
incorrect_predictions = ~correct_predictions

plt.scatter(train_images_pca[correct_predictions, 0], train_images_pca[correct_predictions, 1], c='green', marker='o', edgecolors='k', label='Correct Predictions')
plt.scatter(train_images_pca[incorrect_predictions, 0], train_images_pca[incorrect_predictions, 1], c='red', marker='x', edgecolors='k', label='Incorrect Predictions')

plt.legend()
plt.title('SVM Decision Boundary and Support Vectors with Predictions (PCA)')
plt.show()
