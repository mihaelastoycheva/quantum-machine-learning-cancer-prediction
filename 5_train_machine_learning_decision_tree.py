import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

# === Step 1: Load binary input data ===
X = np.load('X_encoded.npy')   # binary values (0 and 1)
y = np.load('y_labels.npy')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Number of features: {X.shape[1]}, Samples: {len(X)}")

# === Step 2: Define PauliFeatureMap ===
n_qubits = X.shape[1]
feature_map = PauliFeatureMap(
    feature_dimension=n_qubits,
    reps=3,
    entanglement='full',
    paulis=['Z', 'ZZ']
)

# Observables: Z on each qubit
observables = [
    SparsePauliOp.from_list([("I" * i + "Z" + "I" * (n_qubits - i - 1), 1.0)])
    for i in range(n_qubits)
]

estimator = Estimator()

# === Step 3: Extract quantum features using PauliFeatureMap ===
def extract_quantum_features(estimator, circuit_template, observables, X_data):
    circuits = []
    observs = []
    values = []

    for x in X_data:
        for obs in observables:
            circuits.append(circuit_template)  # reuse same parametric circuit
            observs.append(obs)
            values.append(x.tolist())  # each list has values for parameters

    result = estimator.run(
        circuits=circuits,
        observables=observs,
        parameter_values=values
    ).result()

    outputs = np.array(result.values).reshape(len(X_data), len(observables))
    return outputs


print("⚛️ Extracting quantum features...")
X_train_q = extract_quantum_features(estimator, feature_map, observables, X_train)
X_test_q = extract_quantum_features(estimator, feature_map, observables, X_test)

# === Step 4: Train Decision Tree Classifier ===
print("🌳 Training Decision Tree on quantum features...")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_q, y_train)

# === Step 5: Evaluation ===
y_pred = clf.predict(X_test_q)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {accuracy:.2f}")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train_q, orient='h')
plt.title("Distribution of Quantum Features")
plt.xlabel("Feature Value (Z expectation)")
plt.ylabel("Qubit Index")
plt.grid(True)
plt.show()