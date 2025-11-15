📊 ML vs Quantum ML – Model Comparison Project

This project compares the performance of classical machine learning models — Logistic Regression and SVM — with a Quantum Variational Quantum Classifier (VQC).
The goal is to visualize how quantum machine learning performs against traditional ML methods using standard evaluation metrics.

The project generates:

📈 Accuracy comparison

📈 Precision comparison

📈 Recall comparison

📊 Bar graph visualization (scaled to 1)

🧪 A clear comparison between classical ML and Quantum ML models

🔍 What This Project Does

This project:

Loads and preprocesses the dataset

Trains three models:

Logistic Regression

Support Vector Machine (SVM)

Quantum VQC

Computes metrics:

Accuracy

Precision

Recall

Generates a bar graph comparing all three models

Shows how classical ML models outperform quantum VQC on this dataset

📊 What This Project Shows (Results)
Metric	Logistic Regression	SVM	Quantum VQC
Accuracy	0.87	0.85	0.195
Precision	0.8732	0.8514	0.0741
Recall	0.87	0.85	0.195
Quick Summary

Classical ML models (Logistic Regression and SVM) achieve high and consistent performance.

The Quantum VQC performs significantly worse due to quantum circuit limitations, noise, and depth constraints.

The bar graph provides a clear visual comparison between the three models.

🚀 How to Run the Project

This project contains two main parts:

backend/ → Runs all model training, evaluation, and graph generation

frontend/ → Displays results (if applicable) or serves UI components

Follow these steps:

🖥️ Backend Setup (Machine Learning & Quantum ML Code)
1️⃣ Navigate to backend folder
cd backend

2️⃣ Install dependencies
npm install

3️⃣ Start the backend
npm start


or (if using nodemon)

npm run dev

✔️ Backend runs at:
http://localhost:5000

🌐 Frontend Setup (React App)
1️⃣ Navigate to frontend folder
cd frontend

2️⃣ Install dependencies
npm install

3️⃣ Start the frontend
npm start

✔️ Frontend runs at:
http://localhost:3000
