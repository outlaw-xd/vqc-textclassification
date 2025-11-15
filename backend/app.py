import time
import traceback
import numpy as np
import pandas as pd
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# Try Qiskit imports (handle gracefully)
quantum_available = True
_qiskit_error = None
try:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_aer import AerSimulator
    try:
        from qiskit_machine_learning.algorithms.classifiers import VQC
    except:
        from qiskit_machine_learning.algorithms import VQC
except Exception:
    quantum_available = False
    _qiskit_error = traceback.format_exc()

app = Flask(__name__)
CORS(app)

# Global models
log_clf = None
svm_clf = None
vqc = None
vectorizer = None
scaler = None
svd = None
label_encoder = None


@app.route("/")
def home():
    return jsonify({"message": "Quantum + Classical News Classifier Running"})


# ================================
# TRAIN ROUTE
# ================================
@app.route("/train", methods=["POST"])
def train_models():
    global log_clf, svm_clf, vqc, vectorizer, scaler, svd, label_encoder

    quantum_error = None

    try:
        # Load dataset
        df = pd.read_excel("news_headlines_updated.xlsx")
        df.columns = [c.strip().lower() for c in df.columns]

        if "text" not in df.columns:
            for col in df.columns:
                if "headline" in col:
                    df.rename(columns={col: "text"}, inplace=True)
                    break

        if "category" not in df.columns:
            for col in df.columns:
                if "type" in col or "label" in col:
                    df.rename(columns={col: "category"}, inplace=True)
                    break

        df.dropna(subset=["text", "category"], inplace=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        X = df["text"]
        y = df["category"]

        # Improve feature quality
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(X).toarray()

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # =====================
        # 1️⃣ Logistic Regression
        # =====================
        log_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        start = time.time()
        log_clf.fit(X_train_scaled, y_train)
        log_time = time.time() - start

        log_pred = log_clf.predict(X_test_scaled)
        log_acc = accuracy_score(y_test, log_pred)
        log_report = classification_report(y_test, log_pred, output_dict=True)
        log_cm = confusion_matrix(y_test, log_pred).tolist()

        # =====================
        # 2️⃣ SVM
        # =====================
        svm_clf = SVC(kernel="linear", probability=True, class_weight="balanced")
        start = time.time()
        svm_clf.fit(X_train_scaled, y_train)
        svm_time = time.time() - start

        svm_pred = svm_clf.predict(X_test_scaled)
        svm_acc = accuracy_score(y_test, svm_pred)
        svm_report = classification_report(y_test, svm_pred, output_dict=True)
        svm_cm = confusion_matrix(y_test, svm_pred).tolist()

        # =====================
        # 3️⃣ Quantum VQC
        # =====================
        vqc_acc = None
        vqc_report = None
        vqc_cm = None
        vqc_time = None

        if quantum_available:
            try:
                # Safe SVD size
                n_components = min(3, X_train_scaled.shape[1])
                svd = TruncatedSVD(n_components=n_components)
                X_train_svd = svd.fit_transform(X_train_scaled)
                X_test_svd = svd.transform(X_test_scaled)

                feature_dim = X_train_svd.shape[1]
                feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1)
                ansatz = RealAmplitudes(num_qubits=feature_dim, reps=1)
                optimizer = COBYLA(maxiter=30)
                backend = AerSimulator()

                try:
                    vqc = VQC(
                        feature_map=feature_map,
                        ansatz=ansatz,
                        optimizer=optimizer,
                        quantum_instance=backend
                    )
                except:
                    vqc = VQC(
                        feature_map=feature_map,
                        ansatz=ansatz,
                        optimizer=optimizer,
                    )

                start = time.time()
                vqc.fit(X_train_svd, y_train)
                vqc_time = time.time() - start

                q_pred = np.ravel(vqc.predict(X_test_svd)).astype(int)
                vqc_acc = accuracy_score(y_test, q_pred)
                vqc_report = classification_report(y_test, q_pred, output_dict=True)
                vqc_cm = confusion_matrix(y_test, q_pred).tolist()

            except Exception:
                quantum_error = traceback.format_exc()
                vqc = None
        else:
            quantum_error = _qiskit_error

        return jsonify({
            "status": "Training complete",
            "metrics_comparison": {
                "Logistic Regression": {
                    "Accuracy": log_acc,
                    "Precision": log_report["weighted avg"]["precision"],
                    "Recall": log_report["weighted avg"]["recall"],
                    "F1 Score": log_report["weighted avg"]["f1-score"],
                    "Train Time (s)": log_time
                },
                "SVM": {
                    "Accuracy": svm_acc,
                    "Precision": svm_report["weighted avg"]["precision"],
                    "Recall": svm_report["weighted avg"]["recall"],
                    "F1 Score": svm_report["weighted avg"]["f1-score"],
                    "Train Time (s)": svm_time
                },
                "Quantum VQC": {
                    "Accuracy": vqc_acc,
                    "Precision": None if not vqc_report else vqc_report["weighted avg"]["precision"],
                    "Recall": None if not vqc_report else vqc_report["weighted avg"]["recall"],
                    "F1 Score": None if not vqc_report else vqc_report["weighted avg"]["f1-score"],
                    "Train Time (s)": vqc_time
                }
            },
            "confusion_matrices": {
                "Logistic Regression": log_cm,
                "SVM": svm_cm,
                "Quantum VQC": vqc_cm
            },
            "quantum_error": quantum_error
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ================================
# PREDICT ROUTE
# ================================
@app.route("/predict", methods=["POST"])
def predict_news():
    global log_clf, svm_clf, vqc, vectorizer, scaler, svd, label_encoder

    data = request.get_json()
    headline = data.get("headline", "").strip()

    new_vec = vectorizer.transform([headline]).toarray()
    new_scaled = scaler.transform(new_vec)

    log_pred = log_clf.predict(new_scaled)[0]
    svm_pred = svm_clf.predict(new_scaled)[0]

    result = {
        "headline": headline,
        "logistic_regression": label_encoder.inverse_transform([log_pred])[0],
        "svm": label_encoder.inverse_transform([svm_pred])[0],
    }

    if vqc is not None:
        try:
            new_svd = svd.transform(new_scaled)
            q_pred = np.ravel(vqc.predict(new_svd))[0]
            result["quantum_vqc"] = label_encoder.inverse_transform([q_pred])[0]
        except:
            result["quantum_vqc"] = "Quantum prediction failed"

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

