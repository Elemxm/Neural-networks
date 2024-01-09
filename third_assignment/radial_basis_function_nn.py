import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from keras.datasets import cifar10
from keras.utils import to_categorical

class RBFNetwork:
    def __init__(self, num_centers, sigma):
        self.num_centers = num_centers
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _calculate_activations(self, X):
        num_samples = X.shape[0]
        activations = np.zeros((num_samples, self.num_centers))

        for i in range(num_samples):
            for j in range(self.num_centers):
                distance = np.linalg.norm(X[i] - self.centers[j])
                activations[i][j] = np.exp(-distance**2 / (2 * self.sigma**2))

        return activations

    def fit(self, X, y):
        self.centers = X[np.random.choice(X.shape[0], self.num_centers, replace=False)]
        activations = self._calculate_activations(X)

        # Convert y to one-hot encoding
        y_one_hot = to_categorical(y, num_classes=self.num_centers)

        # Compute weights using pseudo-inverse
        self.weights = np.linalg.pinv(activations.T @ activations) @ activations.T @ y_one_hot

    def predict(self, X):
        activations = self._calculate_activations(X)
        return activations @ self.weights

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

center_numbers = [10, 100, 150, 200, 300]
sigma_values = [0.5, 1.0, 2.0, 5.0]

for num_centers in center_numbers:
    for sigma in sigma_values:
        # K-means clustering to get representative centers
        kmeans = KMeans(n_clusters=num_centers, n_init=10, random_state=42)
        kmeans.fit(x_train_flat)
        centers = kmeans.cluster_centers_

        # Initialize and train the RBF network
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
        print(f"The Accuracy for the number of centers {num_centers} and sigma value {sigma} is:", accuracy)
