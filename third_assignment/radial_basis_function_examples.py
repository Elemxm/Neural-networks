import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.datasets import cifar10
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import radial_basis_function_nn

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the training data
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

pca = PCA(0.90)
x_train_flat_pca = pca.fit_transform(x_train_flat)
x_test_flat_pca = pca.transform(x_test_flat)

# Defining the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Initialize and train the RBF network
num_centers = 300
sigma_value = 5.0
rbf_net = radial_basis_function_nn.RBFNetwork(num_centers, sigma_value)

# Standardize features for training data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat_pca)
rbf_net.fit(x_train_scaled, y_train)

# Preprocess the test data
x_test_flat_pca = pca.transform(x_test_flat)
x_test_scaled = scaler.transform(x_test_flat_pca)

# Display images
num_displayed = 15

for i in range(min(num_displayed, len(x_test))):
    true_label = y_test[i]
    
    # Assuming 'rbf_net' has a predict method that returns class predictions
    predicted_label = rbf_net.predict(x_test_scaled[i:i+1])  # Note the slicing to maintain the shape

    # Display the image using matplotlib
    plt.subplot(5, 3, i + 1)
    plt.imshow(x_test[i].reshape(32, 32, 3)) 
    plt.title(f'True: {class_names[true_label[0]]} Predicted: {class_names[np.argmax(predicted_label)]}')
    plt.axis('off')

plt.show()
