import os
import time
import numpy as np
import pandas as pd
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Qiskit imports with fallbacks
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
except:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

try:
    from qiskit_algorithms.optimizers import COBYLA
except:
    try:
        from qiskit_machine_learning.optimizers import COBYLA
    except:
        raise ImportError("COBYLA optimizer not found.")

try:
    from qiskit_aer import AerSimulator
except:
    try:
        from qiskit.providers.aer import AerSimulator
    except:
        AerSimulator = None

try:
    from qiskit.utils import QuantumInstance
except:
    QuantumInstance = None

try:
    from qiskit_machine_learning.algorithms.classifiers import VQC
except:
    try:
        from qiskit_machine_learning.algorithms import VQC
    except:
        raise ImportError("VQC not available.")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app)

# Global models
clf = None
vqc = None
vectorizer = None
scaler = None
svd = None
label_encoder = None

@app.route("/")
def home():
    return jsonify({"message": "Quantum News Classifier Backend Running ✅"})

# ================================
# TRAIN ROUTE WITH METRICS
# ================================
@app.route("/train", methods=["POST"])
def train_models():
    global clf, vqc, vectorizer, scaler, svd, label_encoder

    try:
        df = pd.read_excel("news_headlines.xlsx")
        df.columns = [c.strip().lower() for c in df.columns]

        if "text" not in df.columns:
            text_col = [c for c in df.columns if "headline" in c]
            if not text_col:
                return jsonify({"error": "No 'headline' or 'text' column found"}), 400
            df.rename(columns={text_col[0]: "text"}, inplace=True)

        if "category" not in df.columns:
            cat_col = [c for c in df.columns if "type" in c or "label" in c]
            if not cat_col:
                return jsonify({"error": "No 'category' column found"}), 400
            df.rename(columns={cat_col[0]: "category"}, inplace=True)

        df.dropna(subset=["text", "category"], inplace=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        X = df["text"]
        y = df["category"]

        # Preprocessing
        vectorizer = TfidfVectorizer(max_features=500)
        X_vec = vectorizer.fit_transform(X).toarray()

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Classical Model
        clf = LogisticRegression(max_iter=1000)
        start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        classical_train_time = time.time() - start_time

        y_pred_classical = clf.predict(X_test_scaled)
        classical_acc = accuracy_score(y_test, y_pred_classical)
        classical_report = classification_report(y_test, y_pred_classical, output_dict=True, zero_division=0)
        classical_cm = confusion_matrix(y_test, y_pred_classical).tolist()

        # Extract metrics
        classical_precision = classical_report["weighted avg"]["precision"]
        classical_recall = classical_report["weighted avg"]["recall"]
        classical_f1 = classical_report["weighted avg"]["f1-score"]

        # Quantum Model
        svd = TruncatedSVD(n_components=3, random_state=42)
        X_train_svd = svd.fit_transform(X_train_scaled)
        X_test_svd = svd.transform(X_test_scaled)

        feature_dim = X_train_svd.shape[1]
        feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1)
        ansatz = RealAmplitudes(num_qubits=feature_dim, reps=1)
        optimizer = COBYLA(maxiter=50)

        if AerSimulator is None:
            return jsonify({"error": "AerSimulator not available. Install qiskit-aer."}), 500

        backend = AerSimulator()
        quantum_instance = QuantumInstance(backend, shots=256) if QuantumInstance else None

        vqc_kwargs = {"feature_map": feature_map, "ansatz": ansatz, "optimizer": optimizer}
        if quantum_instance:
            vqc_kwargs["quantum_instance"] = quantum_instance

        vqc = VQC(**vqc_kwargs)
        start_time = time.time()
        vqc.fit(X_train_svd, y_train)
        quantum_train_time = time.time() - start_time

        y_pred_quantum = np.ravel(vqc.predict(X_test_svd)).astype(int)
        quantum_acc = accuracy_score(y_test, y_pred_quantum)
        quantum_report = classification_report(y_test, y_pred_quantum, output_dict=True, zero_division=0)
        quantum_cm = confusion_matrix(y_test, y_pred_quantum).tolist()

        # Extract metrics
        quantum_precision = quantum_report["weighted avg"]["precision"]
        quantum_recall = quantum_report["weighted avg"]["recall"]
        quantum_f1 = quantum_report["weighted avg"]["f1-score"]

        return jsonify({
            "status": "Training complete",
            "metrics_comparison": {
                "Classical Model": {
                    "Accuracy": round(classical_acc, 4),
                    "Precision": round(classical_precision, 4),
                    "Recall": round(classical_recall, 4),
                    "F1 Score": round(classical_f1, 4),
                    "Train Time (s)": round(classical_train_time, 2)
                },
                "Quantum Model": {
                    "Accuracy": round(quantum_acc, 4),
                    "Precision": round(quantum_precision, 4),
                    "Recall": round(quantum_recall, 4),
                    "F1 Score": round(quantum_f1, 4),
                    "Train Time (s)": round(quantum_train_time, 2)
                }
            },
            "classical_confusion_matrix": classical_cm,
            "quantum_confusion_matrix": quantum_cm
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================
# PREDICT ROUTE
# ================================
@app.route("/predict", methods=["POST"])
def predict_news():
    global clf, vqc, vectorizer, scaler, svd, label_encoder

    if clf is None or vqc is None:
        return jsonify({"error": "Models not trained yet. Call /train first."}), 400

    data = request.get_json()
    if not data or "headline" not in data:
        return jsonify({"error": "Missing 'headline' in request JSON"}), 400

    headline = data["headline"].strip()
    if not headline:
        return jsonify({"error": "Empty headline provided"}), 400

    # Transform input
    new_vec = vectorizer.transform([headline]).toarray()
    new_scaled = scaler.transform(new_vec)
    new_svd = svd.transform(new_scaled)

    classical_pred = clf.predict(new_scaled)[0]
    quantum_pred = np.ravel(vqc.predict(new_svd))[0]

    classical_label = label_encoder.inverse_transform([int(classical_pred)])[0]
    quantum_label = label_encoder.inverse_transform([int(quantum_pred)])[0]

    return jsonify({
        "headline": headline,
        "classical_prediction": classical_label,
        "quantum_prediction": quantum_label
    })

# ================================
# RUN SERVER
# ================================
if __name__ == "__main__":
    app.run(debug=True)
