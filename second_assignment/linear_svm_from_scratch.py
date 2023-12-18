from keras import datasets
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import time

# Loading the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

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

#Start Calculating the training time 
start_time = time.time()

#training the SVM classifier with training data
classifier.fit(train_images, train_labels)

#Ending time for training 
end_time = time.time()
print(f'Training time: {end_time - start_time} seconds')

start_time = time.time()
train_images_prediction = classifier.predict(train_images)
end_time = time.time()
print(f'Prediction time train images: {end_time - start_time} seconds')

training_data_accuracy = accuracy_score(train_labels, train_images_prediction)
print(f'the accuracy score on training data is: {training_data_accuracy}')

start_time = time.time()
test_images_prediction = classifier.predict(test_images)
end_time = time.time()
print(f'Prediction time test images: {end_time - start_time} seconds')

testing_data_accuracy = accuracy_score(test_labels, test_images_prediction)
print(f'the accuracy score on testing data is: {testing_data_accuracy}')


num_displayed = 15

for i in range(min(num_displayed, len(test_images))):
    true_label = test_labels[i]
    
    predicted_label = classifier.predict(test_images[i])

    # Display the image using matplotlib
    plt.subplot(5, 3, i + 1)
    plt.imshow(test_images[i].reshape(32, 32, 3))  # Assuming images are 32x32x3
    plt.title(f'True: {class_names[true_label[0]]} Predicted: {class_names[predicted_label[0]]}')
    plt.axis('off')

plt.show()