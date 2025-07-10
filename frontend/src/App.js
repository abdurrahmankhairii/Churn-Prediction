import React, { useState } from 'react';
import logo from './assets/logo.png';
import background from './assets/background.png';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    Age: '',
    Gender: 'Male',
    Tenure: '',
    Usage_Frequency: '',
    Support_Calls: '',
    Payment_Delay: '',
    Subscription_Type: 'Basic',
    Contract_Length: 'Monthly',
    Total_Spend: '',
    Last_Interaction: '',
    model_type: 'logreg'
  });
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setPrediction(null);
    setError(null);
    console.log('Submitting form data:', formData);
    try {
      const response = await fetch('/predict', { // Changed to relative URL for proxy
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      console.log('Fetch response status:', response.status);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('Fetch response data:', data);
      setPrediction(data);
    } catch (error) {
      console.error('Error fetching prediction:', error);
      setError('Failed to fetch prediction. Check console or ensure backend is running.');
      setPrediction({ churn: false, probability: 0 });
    }
  };

  return (
    <div className="container">
      <img src={logo} alt="Logo" className="logo" />
      <h1>Churn Prediction</h1>
      <form onSubmit={handleSubmit} className="form-grid">
        {[
          { label: "Age", name: "Age", type: "number" },
          { label: "Payment Delay", name: "Payment_Delay", type: "number" },
          { label: "Gender", name: "Gender", type: "select", options: ["Male", "Female"] },
          { label: "Subscription Type", name: "Subscription_Type", type: "select", options: ["Basic", "Standard", "Premium"] },
          { label: "Tenure", name: "Tenure", type: "number" },
          { label: "Contract Length", name: "Contract_Length", type: "select", options: ["Monthly", "Quarterly", "Annual"] },
          { label: "Usage Frequency", name: "Usage_Frequency", type: "number" },
          { label: "Total Spend", name: "Total_Spend", type: "number" },
          { label: "Support Calls", name: "Support_Calls", type: "number" },
          { label: "Last Interaction", name: "Last_Interaction", type: "number" },
          { label: "Model", name: "model_type", type: "select", options: ["logreg", "svm", "knn"], labels: ["Logistic Regression", "SVM", "KNN"] }
        ].map(({ label, name, type, options, labels }) => (
          <div className="form-group" key={name}>
            <label>{label}</label>
            {type === "select" ? (
              <select name={name} value={formData[name]} onChange={handleChange}>
                {options.map((opt, idx) => (
                  <option key={opt} value={opt}>
                    {labels ? labels[idx] : opt}
                  </option>
                ))}
              </select>
            ) : (
              <input type={type} name={name} value={formData[name]} onChange={handleChange} required />
            )}
          </div>
        ))}
        <button type="submit">Predict</button>
      </form>

      {error && <div className="error" style={{ color: 'red' }}>{error}</div>}
      {prediction && (
        <div className="result">
          <p><strong>Churn:</strong> {prediction.churn ? 'Yes' : 'No'}</p>
          <p><strong>Probability:</strong> {(prediction.probability * 100).toFixed(2)}%</p>
        </div>
      )}
      <div className="watermark">Abdurrahman Khairi © 2025</div>
    </div>
  );
}

export default App;