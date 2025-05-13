import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# === Step 1: Load and Prepare Data ===
start_time = time.time()

X = np.load('X_encoded.npy')
y = np.load('y_labels.npy')

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 50)
print("📊 Dataset Summary for Training")
print("=" * 50)
print(f"Total examples: {len(X)}")
print(f"Input features per example: {X.shape[1]}")
print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Class distribution (full): {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Class distribution (train): {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Class distribution (test): {dict(zip(*np.unique(y_test, return_counts=True)))}\n")


X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# === Step 2: Define quantum circuit ===
n_qubits = X.shape[1]
input_params = ParameterVector("x", n_qubits)
weight_params = ParameterVector("theta", n_qubits * 2)

qc = QuantumCircuit(n_qubits)

# Input encoding
for i in range(n_qubits):
    qc.h(i)
    qc.ry(input_params[i], i)

# Entanglement
for i in range(n_qubits):
    qc.cx(i, (i + 1) % n_qubits)

# Trainable parameters
depth = 3
weight_params = ParameterVector("theta", 2 * n_qubits * depth)

for layer in range(depth):
    offset = layer * 2 * n_qubits
    for i in range(n_qubits):
        qc.rx(weight_params[offset + 2 * i], i)
        qc.rz(weight_params[offset + 2 * i + 1], i)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)  # more entanglement

# Observable (Z operator on first qubit)
observable = SparsePauliOp.from_list([
    ("Z" * n_qubits, 1)
])


# === Step 3: Define QNN using EstimatorQNN ===
estimator = Estimator()
qnn = EstimatorQNN(
    circuit=qc,
    observables=observable,
    input_params=input_params,
    weight_params=weight_params,
    estimator=estimator
)

model_qnn = TorchConnector(qnn)

# === Step 4: PyTorch wrapper ===
class QuantumCancerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.qnn = model_qnn

    def forward(self, x):
        return torch.sigmoid(self.qnn(x))

model = QuantumCancerClassifier()

# === Step 5: Training ===
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss_history = []

def train(model, X, y, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

print("\n🚀 Training Quantum Neural Network...")
train(model, X_train_torch, y_train_torch, epochs=50)
print("✅ Training complete!")

# Draw learning curve
plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker='o')
plt.title("Quantum Model Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# === Step 6: Evaluation ===
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_torch)
    y_pred_labels = (y_pred_probs > 0.5).float()


print("Predicted classes:", torch.unique(y_pred_labels))
print("Train label distribution:", np.bincount(y_train))
print("Test label distribution:", np.bincount(y_test))

print("Example predictions:", y_pred_probs[:10].flatten())

accuracy = accuracy_score(y_test_torch.numpy(), y_pred_labels.numpy())
print(f"\n🎯 Test Accuracy: {accuracy:.2f}\n")
print("📋 Classification Report:")
print(classification_report(y_test_torch.numpy(), y_pred_labels.numpy(), zero_division=0))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱ Total execution time: {elapsed_time:.2f} seconds")
