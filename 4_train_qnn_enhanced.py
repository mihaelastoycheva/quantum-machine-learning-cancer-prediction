import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import PowerTransformer

from imblearn.over_sampling import SMOTE

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from qiskit_algorithms.optimizers import SPSA


# =================================================================
# 1. Data loading and preparation
# =================================================================
def load_data():
    X = np.load('X_encoded.npy')  # shape: (798, 12)
    y = np.load('y_labels.npy')  # shape: (798,)

    print(f"📊 Training data: {X.shape[0]} examples")
    print(f"🔢 Distribution of classes: {np.bincount(y)} (0: healthy, 1: mutation)")
    return X, y


# =================================================================
# 2. Pre-processing
# =================================================================
def preprocess_data(X, y, test_size=0.2):
    # Normalization
    scaler = PowerTransformer(method='yeo-johnson')
    X_scaled = scaler.fit_transform(X)

    # Data separation BEFORE balancing
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    # Balancing training data only
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print("\n🔧 Data after processing:")
    print(f"- Training: {X_train_bal.shape} (balanced)")
    print(f"- Test: {X_test.shape}")

    return X_train_bal, X_test, y_train_bal, y_test


# =================================================================
# 3. Quantum Model Configuration
# =================================================================
def build_quantum_model(n_qubits=4):
    """Creates a quantum classifier with logistic regression"""
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=5,
        entanglement='linear',
        insert_barriers=True
    )

    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=2,
        entanglement='circular',
        insert_barriers=True
    )

    # Initialisation on Sampler with GPU acceleration (only for Linux)
    try:
        sampler = Sampler(backend=AerSimulator(method='statevector'))
    except:
        sampler = Sampler()

    optimizer = SPSA(
        maxiter=300,
        learning_rate=0.1,
        perturbation=0.1,
    )

    model = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        sampler=sampler,
        initial_point=np.random.rand(ansatz.num_parameters) * 2 * np.pi
    )

    # Visualisation of quantum circuit
    fm_bound = feature_map.assign_parameters(np.zeros(feature_map.num_parameters))
    ansatz_bound = ansatz.assign_parameters(np.zeros(ansatz.num_parameters))
    full_circuit = fm_bound.compose(ansatz_bound)
    full_circuit.decompose().draw(output='mpl')
    plt.show()

    return model


# =================================================================
# 4. Evaluation and visualisation
# =================================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n🎯 Accurancy: {accuracy:.2%}")
    print(f"📊 ROC AUC: {roc_auc:.2%}")
    print("\n📝 Classification report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Mutation']))

    # =================================================================


# Main function
# =================================================================
def main():
    # 1. Loading data
    X, y = load_data()

    # 2. Pre-processing
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # 3. Selection of number of qubits (optimal: 4 for 12 bits)
    n_qubits = min(4, X_train.shape[1])
    print(f"\n⚛️ Used {n_qubits} qubits of {X_train.shape[1]} features")

    # 4. Setting up and training the model
    model = build_quantum_model(n_qubits)
    print("\n🌀 Training model ...")
    model.fit(X_train[:, :n_qubits], y_train)

    # 5. Evaluation
    evaluate_model(model, X_test[:, :n_qubits], y_test)


if __name__ == "__main__":
    main()