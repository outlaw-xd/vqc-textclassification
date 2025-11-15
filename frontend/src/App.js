import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [headline, setHeadline] = useState("");
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  const num = (v) => Number(v) || 0;

  // Train all models
  const handleTrain = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/train");
      setMetrics(res.data.metrics_comparison);
    } catch (err) {
      alert("Training failed! Backend might be offline.");
    } finally {
      setLoading(false);
    }
  };

  // Predict headline category
  const handlePredict = async () => {
    if (!headline.trim()) {
      alert("Please enter a news headline!");
      return;
    }
    setLoading(true);

    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", { headline });
      setResult(res.data);
    } catch (err) {
      alert("Prediction failed!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Quantum vs Classical News Classifier</h1>
      <p>Compare Logistic Regression, SVM, and Quantum VQC.</p>

      {/* Input */}
      <div className="controls">
        <textarea
          placeholder="Enter a news headline..."
          value={headline}
          onChange={(e) => setHeadline(e.target.value)}
        />
        <div className="buttons">
          <button onClick={handleTrain} disabled={loading}>
            {loading ? "Training..." : "Train Models"}
          </button>

          <button onClick={handlePredict} disabled={loading}>
            {loading ? "Predicting..." : "Predict Headline"}
          </button>
        </div>
      </div>

      {/* Metrics Table */}
      {metrics && (
        <div className="metrics">
          <h2>Model Performance Comparison</h2>

          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Logistic Regression</th>
                <th>SVM</th>
                <th>Quantum VQC</th>
              </tr>
            </thead>

            <tbody>
              <tr>
                <td>Accuracy</td>
                <td>{num(metrics["Logistic Regression"].Accuracy).toFixed(4)}</td>
                <td>{num(metrics["SVM"].Accuracy).toFixed(4)}</td>
                <td>{num(metrics["Quantum VQC"].Accuracy).toFixed(4)}</td>
              </tr>

              <tr>
                <td>Precision</td>
                <td>{num(metrics["Logistic Regression"].Precision).toFixed(4)}</td>
                <td>{num(metrics["SVM"].Precision).toFixed(4)}</td>
                <td>{num(metrics["Quantum VQC"].Precision).toFixed(4)}</td>
              </tr>

              <tr>
                <td>Recall</td>
                <td>{num(metrics["Logistic Regression"].Recall).toFixed(4)}</td>
                <td>{num(metrics["SVM"].Recall).toFixed(4)}</td>
                <td>{num(metrics["Quantum VQC"].Recall).toFixed(4)}</td>
              </tr>

              <tr>
                <td>F1 Score</td>
                <td>{num(metrics["Logistic Regression"]["F1 Score"]).toFixed(4)}</td>
                <td>{num(metrics["SVM"]["F1 Score"]).toFixed(4)}</td>
                <td>{num(metrics["Quantum VQC"]["F1 Score"]).toFixed(4)}</td>
              </tr>

              <tr>
                <td>Train Time (s)</td>
                <td>{num(metrics["Logistic Regression"]["Train Time (s)"]).toFixed(2)}</td>
                <td>{num(metrics["SVM"]["Train Time (s)"]).toFixed(2)}</td>
                <td>{num(metrics["Quantum VQC"]["Train Time (s)"]).toFixed(2)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Prediction Results */}
      {result && (
        <div className="results">
          <h2>Prediction Results</h2>
          <p><strong>Headline:</strong> {result.headline}</p>

          <p><strong>Logistic Regression:</strong> {result.logistic_regression}</p>
          <p><strong>SVM:</strong> {result.svm}</p>
          <p><strong>Quantum VQC:</strong> {result.quantum_vqc}</p>
        </div>
      )}
    </div>
  );
}

export default App;

