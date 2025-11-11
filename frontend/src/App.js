import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [headline, setHeadline] = useState("");
  const [result, setResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);

  // Train models and fetch metrics
  const handleTrain = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://127.0.0.1:5000/train");
      setMetrics(res.data.metrics_comparison); // Extract metrics_comparison object
    } catch (err) {
      alert("Training failed! Make sure Flask backend is running.");
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
      alert("Prediction failed! Make sure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1> Quantum vs Classical News Classifier</h1>
      <p>Compare predictions and metrics between classical and quantum models.</p>

      {/* Input + Buttons */}
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
                <th>Classical Model</th>
                <th>Quantum Model</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Accuracy</td>
                <td>{metrics["Classical Model"].Accuracy.toFixed(4)}</td>
                <td>{metrics["Quantum Model"].Accuracy.toFixed(4)}</td>
              </tr>
              <tr>
                <td>Precision</td>
                <td>{metrics["Classical Model"].Precision.toFixed(4)}</td>
                <td>{metrics["Quantum Model"].Precision.toFixed(4)}</td>
              </tr>
              <tr>
                <td>Recall</td>
                <td>{metrics["Classical Model"].Recall.toFixed(4)}</td>
                <td>{metrics["Quantum Model"].Recall.toFixed(4)}</td>
              </tr>
              <tr>
                <td>F1 Score</td>
                <td>{metrics["Classical Model"]["F1 Score"].toFixed(4)}</td>
                <td>{metrics["Quantum Model"]["F1 Score"].toFixed(4)}</td>
              </tr>
              <tr>
                <td>Train Time (s)</td>
                <td>{metrics["Classical Model"]["Train Time (s)"].toFixed(2)}</td>
                <td>{metrics["Quantum Model"]["Train Time (s)"].toFixed(2)}</td>
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
          <p><strong>Classical Model:</strong> {result.classical_prediction}</p>
          <p><strong>Quantum Model:</strong> {result.quantum_prediction}</p>
        </div>
      )}
    </div>
  );
}

export default App;
