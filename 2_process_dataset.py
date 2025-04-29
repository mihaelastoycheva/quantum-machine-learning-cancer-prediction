import pandas as pd
import numpy as np


# Function to extract a 6-base fragment centered around the mutation
def extract_mutation_centered_chunk(sequence, mutation_pos_in_window, window=6):
    half_window = window // 2
    start = max(mutation_pos_in_window - half_window, 0)
    end = start + window
    if end > len(sequence):
        end = len(sequence)
        start = end - window
    return sequence[start:end]


# Function to encode a DNA sequence into a binary vector
def encode_dna_to_binary_vector(seq):
    binary_data = ''
    for nucleotide in seq:
        if nucleotide == 'A':
            binary_data += '00'
        elif nucleotide == 'C':
            binary_data += '01'
        elif nucleotide == 'G':
            binary_data += '10'
        elif nucleotide == 'T':
            binary_data += '11'
        else:
            binary_data += '00'  # Default for unknown characters
    return [int(bit) for bit in binary_data]


# Load the dataset
classical_dataset = pd.read_csv('classical_dataset.csv')

# Prepare containers for features (x) and labels (y)
X = []
y = []

# Settings
full_window_size = 100  # The original sequence length
chunk_size = 6  # We want to extract 6-base fragments
mutation_center_pos = 50  # Mutation is approximately at the center (index 50)

# Process each record
for idx, row in classical_dataset.iterrows():
    sequence = row['sequence']
    label = row['label']

    # Extract a chunk centered around the mutation position
    chunk = extract_mutation_centered_chunk(sequence, mutation_pos_in_window=mutation_center_pos, window=chunk_size)

    # Encode the chunk into a binary vector
    binary_vector = encode_dna_to_binary_vector(chunk)

    X.append(binary_vector)
    y.append(label)

# Convert the lists into NumPy arrays
X = np.array(X)
y = np.array(y)

print("✅ Data preparation completed successfully!")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Save the preprocessed data for future use
np.save('X_encoded.npy', X)
np.save('y_labels.npy', y)
