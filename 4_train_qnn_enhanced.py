import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Зареждане на данни
X = np.load('X_encoded.npy')
y = np.load('y_labels.npy')

# 2. Предварителна обработка
pca = PCA(n_components=4)  # Намаляваме до 4 компонента за съответствие с qubits
X_pca = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 3. Квантова конфигурация
n_qubits = X_pca.shape[1]  # Брой qubits = брой PCA компоненти

# Опростен ZZFeatureMap вместо NLocal
feature_map = ZZFeatureMap(
    feature_dimension=n_qubits,
    reps=2,
    entanglement='linear'
)

# 4. Квантово ядро
fidelity = ComputeUncompute(sampler=Sampler())
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# 5. Дефиниране на GridSearchCV
param_grid = {'C': [0.1, 1, 10]}  # Тестване на различни стойности за регуларизация
qsvm = GridSearchCV(
    SVC(kernel=quantum_kernel.evaluate),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1  # Паралелно изчисление (ако имате многоядрен процесор)
)

# 6. Обучение
print("🔍 Starting Grid Search...")
qsvm.fit(X_train, y_train)

# 7. Оценка с най-добрите параметри
print("\n🔮 Best SVM parameters:", qsvm.best_params_)
y_pred = qsvm.predict(X_test)
print(classification_report(y_test, y_pred))