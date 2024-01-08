# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
# from scipy.spatial.distance import cdist
# from sklearn.linear_model import LogisticRegression  # Change to LogisticRegression for classification

# X, y = make_classification(n_samples=300, n_features=2, n_classes=2, n_clusters_per_class=2, n_redundant=0, random_state=42)

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
# plt.title("Sample Dataset")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# def gaussian_rbf(x, center, sigma):
#     return np.exp(-cdist(x, center, 'sqeuclidean') / (2 * sigma**2))

# n_centers = 10  # Number of RBF centers
# center_indices = np.random.choice(X.shape[0], n_centers, replace=False)
# rbf_centers = X[center_indices]
# rbf_width = 1.0

# def rbf_layer(X, rbf_centers, rbf_width):
#     return gaussian_rbf(X, rbf_centers, rbf_width)

# def rbfn_predict(X, rbf_centers, rbf_width, weights):
#     rbf_outputs = rbf_layer(X, rbf_centers, rbf_width)
#     return rbf_outputs @ weights  # Fix the return statement

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# rbf_outputs_train = rbf_layer(X_train, rbf_centers, rbf_width)

# # Perform logistic regression for classification
# lr = LogisticRegression()
# lr.fit(rbf_outputs_train, y_train)

# # Make predictions on the test set
# rbf_outputs_test = rbf_layer(X_test, rbf_centers, rbf_width)
# y_pred = lr.predict(rbf_outputs_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")





import numpy as np

class RBFNetwork:
    def __init__(self, num_centers, sigma):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _calculate_activations(self, X):
        num_samples = X.shape[0]
        num_features = X.shape[1]
        activations = np.zeros((num_samples, self.num_centers))

        for i in range(num_samples):
            for j in range(self.num_centers):
                distance = np.linalg.norm(X[i] - self.centers[j])
                activations[i][j] = np.exp(-distance**2 / (2 * self.sigma**2))

        return activations

    def fit(self, X, y):
        self.centers = X[np.random.choice(X.shape[0], self.num_centers, replace=False)]
        activations = self._calculate_activations(X)

        self.weights = np.linalg.pinv(activations.T @ activations) @ activations.T @ y

    def predict(self, X):
        activations = self._calculate_activations(X)
        return activations @ self.weights

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load CIFAR-10 dataset (using TensorFlow as an example)
from keras.datasets import cifar10

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

# Initialize and train the RBF network
num_centers = 100  # You may need to experiment with the number of centers
sigma = 1.0
rbf_net = RBFNetwork(num_centers, sigma)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
rbf_net.fit(x_train_scaled, y_train)

# Predict on the test set
x_test_scaled = scaler.transform(x_test_flat)
predictions = rbf_net.predict(x_test_scaled)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, predicted_labels)
print("Accuracy:", accuracy)