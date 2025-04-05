# 🛡️ Advanced Fraud Detection System

[![Python 3.10](https://img.shields.io/badge/Pytww.python.org/downloads/r://img.shields.io/badge/scikit--learn-1.2.2-orange.svg//img.shields.io/badge/Flaflask.palletsprojects.commg.shields.io/badge/SHAP-0.46.0-red.svgent](https://img.shields.io/badge/Deployed-Render-9cf.svect presents a fraud detection system using a Random Forest classifier with explainable AI features, trained on the Credit Card Fraud Detection Dataset 2023 from Kaggle. The system achieves **99% recall for fraud detection** with a high precision rate, providing both accurate predictions and transparent decision insights through SHAP value analysis.

## 📊 Key Performance Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| Recall (Fraud Class) | 0.99 | Detects 99% of fraudulent transactions |
| Precision (Fraud Class) | 0.67 | High precision despite class imbalance |
| ROC-AUC | ~0.99 | Excellent discrimination ability |
| False Negatives | 4 out of 574 | Only 4 fraudulent transactions missed |

## ✨ Key Features

- **Robust ML Pipeline**: Random Forest classifier optimized for imbalanced credit card fraud data
- **Explainable AI**: SHAP (SHapley Additive exPlanations) for transparent model predictions
- **Production-Ready API**: Flask-based REST API with comprehensive error handling
- **Live Deployment**: Hosted on Render with real-time prediction capabilities
- **Imbalance Handling**: Specialized techniques to address the 99:1 class imbalance

## 🧠 System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ Transaction │    │ ML Prediction │    │ SHAP Analysis │
│   Request   │───▶│     Model     │───▶│ & Explanation │
└─────────────┘    └──────────────┘    └───────────────┘
                          │                     │
                          ▼                     ▼
                   ┌──────────────────────────────┐
                   │      JSON Response with      │
                   │   Prediction & Explanation   │
                   └──────────────────────────────┘
```

## 🚀 API Usage

### Local Testing

```bash
# Start the server
python app.py

# Make a prediction request
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [-0.260648, -0.469648, 2.496266, -0.083724, 0.129681, 0.732898, 0.519014, -0.130006, 0.727159, 0.637735, -1.289146, 0.507876, 0.019821, 1.443803, 0.151603, -0.339666, -0.673666, -0.117375, 0.450852, -0.114963, -0.110552, 0.217606, -0.134794, 0.165959, 0.126280, -0.434824, -0.081230, -0.151045, 17982.10]}' \
http://127.0.0.1:5000/predict
```

### Production API

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [-0.260648, -0.469648, 2.496266, -0.083724, 0.129681, 0.732898, 0.519014, -0.130006, 0.727159, 0.637735, -1.289146, 0.507876, 0.019821, 1.443803, 0.151603, -0.339666, -0.673666, -0.117375, 0.450852, -0.114963, -0.110552, 0.217606, -0.134794, 0.165959, 0.126280, -0.434824, -0.081230, -0.151045, 17982.10]}' \
https://fraud-detection-ai.onrender.com/predict
```

### Sample Response

```json
{
  "fraud": 1,
  "probability": 0.87,
  "explanation": "Prediction influenced by: V3 (value: 2.50, SHAP: 0.2345) increased the likelihood of fraud; V11 (value: -1.29, SHAP: 0.1876) increased the likelihood of fraud"
}
```

## 📋 Technical Implementation Details

### Model Selection Rationale

The Random Forest classifier was chosen for this project due to its:

- **Robustness to outliers** common in financial transaction data
- **Ability to handle non-linear relationships** between features
- **Natural handling of class imbalance** with class_weight parameter
- **Interpretability** when combined with SHAP explanations
- **Superior performance** in comparative testing against logistic regression

### Handling Extreme Class Imbalance

The dataset exhibits a 99:1 class imbalance (non-fraud:fraud), which was addressed through:

- **Class weighting** during model training
- **Threshold optimization** (0.03) to improve recall for the minority class
- **PR-AUC** as the primary evaluation metric instead of accuracy

### SHAP Implementation

The system implements SHAP (SHapley Additive exPlanations) to interpret model predictions:

- **TreeExplainer** for efficient computation of SHAP values from Random Forest
- **Feature contribution analysis** highlighting which transaction characteristics led to the fraud prediction
- **Multiple output format handling** to ensure robust explanation generation

## 🔧 Setup & Installation

```bash
# Clone the repository
git clone https://github.com/AjmalMIITM/fraud-detection-ai.git
cd fraud-detection-ai

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
python app.py
```

## 🗂️ Project Structure

```
├── app.py                 # Flask API for model serving
├── fraud_detection.ipynb  # Model training & evaluation notebook
├── rf_model.pkl           # Trained Random Forest model
├── requirements.txt       # Project dependencies
├── proposal.md            # Project proposal with details
└── README.md              # Project documentation
```

## 📊 Dataset

The Credit Card Fraud Detection Dataset 2023 from Kaggle features:
- 568,630 transactions with anonymized numerical features V1-V28
- Equal distribution of fraud and non-fraud in the original dataset, rebalanced to reflect real-world imbalance
- PCA-transformed features for data privacy

## 🔮 Future Enhancements

- **Model Visualization Pipeline**: Add interactive SHAP visualizations for feature importance
- **Enhanced API Security**: Implement token-based authentication and rate limiting
- **Model Monitoring System**: Add drift detection and retraining triggers
- **Gradio Dashboard**: Create a user-friendly interface for non-technical stakeholders
- **Documentation Expansion**: Add API specification and detailed model documentation

## 👨‍💻 Author

**Ajmal M**  
IIT Madras  
[Email](mailto:ajmal@iitm.ac.in) | [GitHub](https://github.com/AjmalMIITM)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
