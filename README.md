📰 Text Classification Using Classical ML & Quantum VQC

This project classifies news articles into five categories:

🎭 Entertainment
🏅 Sports
🏛️ Politics
💼 Business
🖥️ Technology

The classification is performed using both classical machine learning and quantum machine learning, allowing a clear performance comparison between the two paradigms.
This project Loads and preprocesses a labeled news dataset Converts text into numerical features using NLP techniques (e.g., TF-IDF) and Trains two classical models and one quantum model as:

Logistic Regression
Support Vector Machine (SVM)
Variational Quantum Classifier (VQC)

Computes evaluation metrics:

Accuracy
Precision
Recall
F1-Score
Training Time

Output :
Evalution Matrix
News Headline Category classification

🚀 How to Run the Project

The project has two main components:

backend/ → Runs ML models, quantum VQC, predictions, and metrics
frontend/ → React UI to display results and graphs

🖥️ Backend Setup (ML + Quantum VQC)

1️⃣ Navigate to the backend folder
cd backend

2️⃣ Install backend dependencies
npm install

3️⃣ Start backend server
npm start

or

npm run dev

✔ Backend runs on:
http://localhost:5000



🌐 Frontend Setup (React Interface)

1️⃣ Navigate to the frontend folder
cd frontend

2️⃣ Install frontend dependencies
npm install

3️⃣ Start the frontend
npm start

✔ Frontend runs on:
http://localhost:3000
