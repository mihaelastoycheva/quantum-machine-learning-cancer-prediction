import numpy as np

# ------------------------------- Print in console

# Load X file (binary encoded sequences)
X = np.load('X_encoded.npy')

# Load y file (labels)
y = np.load('y_labels.npy')

# Check the data
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("X entries:\n", X[:20])
print("y labels:\n", y[:10])

# ------------------------------- Save in file

'''

# Load the .npy files
X = np.load('X_encoded.npy')
y = np.load('y_labels.npy')

# Open a text file for writing
with open('dataset_output.txt', 'w') as f:
    f.write("X_encoded (Input Features):\n")

    # Write X data
    for i, row in enumerate(X):
        row_str = ' '.join(map(str, row))
        f.write(f"Sample {i}: {row_str}\n")

    f.write("\n")
    f.write("y_labels (Output Labels):\n")

    # Write y data
    for i, label in enumerate(y):
        f.write(f"Sample {i}: {label}\n")

print("✅ Data successfully written to 'dataset_output.txt'")

'''