import matplotlib.pyplot as plt
import numpy as np

# Read data from the file
with open('second_assignment\svm_results.txt', 'r') as file:
    data = file.readlines()

# Initialize lists to store extracted data
C_values, gamma_values, accuracy_train, accuracy_test = [], [], [], []

# Extract data from each line
for line in data:
    if "SVC Linear Classification results" in line:
        # Extract C and gamma values
        C, gamma = map(float, line.split("for C = ")[1].split(" and gamma = ")[0:2])
        C_values.append(C)
        gamma_values.append(gamma)

    # Extract accuracy values
    if "The accuracy score on training data is:" in line:
        accuracy_train.append(float(line.split(":")[1]))

    if "The accuracy score on testing data is:" in line:
        accuracy_test.append(float(line.split(":")[1]))

# Convert lists to numpy arrays for better manipulation
C_values = np.array(C_values)
gamma_values = np.array(gamma_values)
accuracy_train = np.array(accuracy_train)
accuracy_test = np.array(accuracy_test)

# Define a color map for discrete gamma values
gamma_colors = {0.001: 'red', 0.01: 'green', 0.1: 'blue', 1: 'purple', 10: 'orange', 100: 'brown'}

# Get unique gamma values
unique_gamma_values = np.unique(gamma_values)

def ploting_func(unique_gamma_values, C_values, accuracy, gamma_colors, type_of_data):
# Plotting Training Data Accuracy
    plt.figure(figsize=(12, 6))
    for gamma in unique_gamma_values:
        # Filter data for each gamma value
        mask = (gamma_values == gamma)
        sorted_indices = np.argsort(C_values[mask])
        plt.plot(C_values[mask][sorted_indices], accuracy[mask][sorted_indices],
                'o-', label=f'{type_of_data} Data (Gamma={gamma})', color=gamma_colors[gamma])

    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Accuracy')
    plt.xscale('log')  # Logarithmic scale for better visualization
    plt.title(f'SVM Linear Classification {type_of_data} Data Accuracy')
    plt.legend()
    plt.show()

type_of_data = ['Training', 'Testing']
ploting_func(unique_gamma_values, C_values, accuracy_train, gamma_colors, type_of_data[0])
ploting_func(unique_gamma_values, C_values, accuracy_test, gamma_colors, type_of_data[1])