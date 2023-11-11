import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score

# Load CIFAR-10 dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
file = r'C:\Users\Elena\Downloads\cifar-10-batches-py\data_batch_1'
data_batch_1 = unpickle(file)

# Categorize the data into images and labels
data = data_batch_1[b'data']  # Assuming 'data' is the key for images
labels = data_batch_1[b'labels']  # Assuming 'labels' is the key for labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 1-NN Classifier
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(X_train, y_train)
knn_1_predictions = knn_1.predict(X_test)
accuracy_1nn = accuracy_score(y_test, knn_1_predictions)

# 3-NN Classifier
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(X_train, y_train)
knn_3_predictions = knn_3.predict(X_test)
accuracy_3nn = accuracy_score(y_test, knn_3_predictions)

# Nearest Center Classifier
nearest_center = NearestCentroid()
nearest_center.fit(X_train, y_train)
nearest_center_predictions = nearest_center.predict(X_test)
accuracy_nearest_center = accuracy_score(y_test, nearest_center_predictions)

# Print accuracies
print("Accuracy of 1-NN Classifier: {:.2f}%".format(accuracy_1nn * 100))
print("Accuracy of 3-NN Classifier: {:.2f}%".format(accuracy_3nn * 100))
print("Accuracy of Nearest Center Classifier: {:.2f}%".format(accuracy_nearest_center * 100))
