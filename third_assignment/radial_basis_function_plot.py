import matplotlib.pyplot as plt
import numpy as np

# Read data from the file
with open('rbf_results.txt', 'r') as file:
    data = file.readlines()

# Initialize lists to store extracted data
num_centers_values, sigma_values, training_time, accuracy_train, accuracy_test = [], [], [], [], []

# Extract data from each line
for line in data:
    if "Training time:" in line:
        training_time.append(float(line.split(":")[1].split(" seconds")[0]))

    if "The Training Accuracy" in line:
        accuracy_train.append(float(line.split(":")[1]))

    if "The Testing Accuracy" in line:
        accuracy_test.append(float(line.split(":")[1]))

    if "number of centers" in line and "sigma value" in line:
        # Extract both num_centers and sigma using a more robust method
        num_centers, sigma = map(float, line.split("number of centers")[1].split("and sigma value ")[0:2])
        num_centers_values.append(num_centers)
        sigma_values.append(sigma)

# Convert lists to numpy arrays for better manipulation
num_centers_values = np.array(num_centers_values)
sigma_values = np.array(sigma_values)
training_time = np.array(training_time)
accuracy_train = np.array(accuracy_train)
accuracy_test = np.array(accuracy_test)

# Define a color map for discrete sigma values
sigma_colors = {'0.1': 'red', '0.5': 'green', '1.0': 'blue', '2.0': 'purple', '4.0': 'orange', '5.0': 'brown'}

# Get unique sigma values
unique_sigma_values = np.unique(sigma_values)

def ploting_func(unique_sigma_values, num_centers_values, accuracy, sigma_colors, type_of_data):
    # Plotting Training Data Accuracy
    plt.figure(figsize=(12, 6))
    for sigma in unique_sigma_values:
        # Filter data for each sigma value
        mask = (sigma_values == sigma)
        sorted_indices = np.argsort(num_centers_values[mask])
        plt.plot(num_centers_values[mask][sorted_indices], accuracy[mask][sorted_indices],
                 'o-', label=f'{type_of_data} Data (Sigma={sigma})', color=sigma_colors[str(sigma)])

    plt.xlabel('Number of Centers')
    plt.ylabel('Accuracy')
    plt.title(f'RBF Network Classification {type_of_data} Data Accuracy')
    plt.legend()
    plt.show()

ploting_func(unique_sigma_values, num_centers_values, accuracy_train, sigma_colors, 'Training')
ploting_func(unique_sigma_values, num_centers_values, accuracy_test, sigma_colors, 'Testing')