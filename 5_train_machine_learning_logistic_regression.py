import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

# === Step 1: Load binary input data ===
X = np.load('X_encoded.npy')   # binary values (0 and 1)
y = np.load('y_labels.npy')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Number of features: {X.shape[1]}, Samples: {len(X)}")

# === Step 2: Improved Quantum circuit for binary input ===
n_qubits = X.shape[1]
input_params = ParameterVector("x", n_qubits)

qc = QuantumCircuit(n_qubits)

# Enhanced encoding: multiple rotations + full entanglement
for i in range(n_qubits):
    qc.h(i)
    qc.rx(np.pi * input_params[i], i)

for i in range(n_qubits - 1):
    qc.cx(i, i + 1)

# Observables: Z on each qubit
observables = [
    SparsePauliOp.from_list([("I" * i + "Z" + "I" * (n_qubits - i - 1), 1.0)])
    for i in range(n_qubits)
]

estimator = Estimator()

# === Step 3: Efficient batch quantum feature extraction ===
def extract_quantum_features(estimator, circuit, observables, X_data):
    circuits = []
    observs = []
    values = []
    for x in X_data:
        for obs in observables:
            circuits.append(circuit)
            observs.append(obs)
            values.append(x.tolist())
    result = estimator.run(circuits=circuits, observables=observs, parameter_values=values).result()
    outputs = np.array(result.values).reshape(len(X_data), len(observables))
    return outputs

print("⚛️ Extracting quantum features...")
X_train_q = extract_quantum_features(estimator, qc, observables, X_train)
X_test_q = extract_quantum_features(estimator, qc, observables, X_test)

# === Step 4: Logistic regression ===
print("🚀 Training logistic regression on quantum features...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_q, y_train)

# === Step 5: Evaluation ===
y_pred = clf.predict(X_test_q)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.2f}")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

plt.figure(figsize=(10, 6))
sns.boxplot(data=X_train_q, hue=y_train[:len(X_train_q)])
plt.title("Quantum Features per Class")
plt.xlabel("Qubit")
plt.ylabel("Expectation Value")
plt.show()
