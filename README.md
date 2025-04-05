# 🛡️ Advanced Fraud Detection System (Explainable AI)

> **99% Recall • SHAP Explanations • Render Deployed API**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.5-lightgrey.svg)](https://flask.palletsprojects.com/)
[![SHAP](https://img.shields.io/badge/SHAP-0.46.0-red.svg)](https://github.com/slundberg/shap)
[![Deployed on Render](https://img.shields.io/badge/Deployed-Render-9cf.svg)](https://fraud-detection-ai.onrender.com/)

A production-ready, end-to-end **Fraud Detection System** trained on real-world transaction data and enriched with **Explainable AI** using SHAP. Built to detect fraudulent transactions with **high recall and interpretability**, making it suitable for financial institutions and risk analytics.

---

## 🚀 Results at a Glance

| Metric                    | Score | Description                                  |
|---------------------------|-------|----------------------------------------------|
| **Recall (Fraud Class)**  | 0.99  | Captures 99% of all fraudulent transactions  |
| **Precision (Fraud Class)** | 0.67  | Optimized precision despite imbalance         |
| **ROC-AUC**               | ~0.99 | Outstanding model discrimination              |
| **False Negatives**       | 4/574 | Extremely low missed fraud cases              |

---

## 💡 Features

✅ **Optimized Random Forest Classifier**  
✅ **SHAP for Transparent Explanations**  
✅ **Flask API for Real-Time Predictions**  
✅ **Deployed on Render (Free Tier)**  
✅ **Threshold Tuning for Class Imbalance**  
✅ **High Accuracy with Model Interpretability**  

---

## 🧠 System Architecture

```
User Input → ML Model → Prediction → SHAP Explainer → JSON Response
```

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ Transaction │ → │ Fraud Model  │ → │ SHAP Explainer │
└─────────────┘    └──────────────┘    └───────────────┘
                             ↓
                JSON Output: Prediction + Why?
```

---

## 🌐 Live Demo (API)

### ✅ Example Usage:

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [-0.260648, -0.469648, 2.496266, ..., 17982.10]}' \
https://fraud-detection-ai.onrender.com/predict
```

### ✅ Sample Response:

```json
{
  "fraud": 1,
  "probability": 0.87,
  "explanation": "Prediction influenced by: V3 (value: 2.50, SHAP: 0.2345), V11 (value: -1.29, SHAP: 0.1876)"
}
```

---

## 📊 Model Strategy

### 📌 Why Random Forest?

- Handles **outliers** and **non-linear features** well  
- Naturally supports **imbalanced classification** via class weights  
- Works great with SHAP’s `TreeExplainer`  
- Outperformed other models in benchmark tests  

### 🧪 Imbalance Handling

- **Class Weight Balancing**  
- **Threshold Shift to 0.03** for higher fraud sensitivity  
- **PR-AUC as Main Metric** instead of Accuracy  

---

## 🧰 SHAP for Explainable AI

- **TreeExplainer** used for efficient SHAP computation  
- Each prediction comes with **human-readable insights**  
- Transparent feature contribution for every decision  

---

## 🔧 Setup & Run Locally

```bash
git clone https://github.com/AjmalMIITM/fraud-detection-ai.git
cd fraud-detection-ai

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

Then open: `http://localhost:5000`

---

## 📁 Project Structure

```
📂 fraud-detection-ai
├── app.py                 → Flask API endpoint
├── fraud_detection.ipynb  → Model training & SHAP analysis
├── rf_model.pkl           → Trained model
├── requirements.txt       → Dependency list
├── proposal.md            → System planning document
└── README.md              → You’re here!
```

---

## 📂 Dataset Info

- Source: **Kaggle (2023)** Credit Card Fraud Dataset  
- PCA-transformed, anonymized features (V1-V28)  
- Real-world class imbalance simulated (99:1)  
- Total Transactions: ~569,000+

---

## 🌱 Future Work

🔒 Add API Authentication & Rate Limiting  
📊 Interactive SHAP Dashboard via Gradio  
📉 Model Drift Detection & Auto-Retraining  
📚 Expanded Documentation with Swagger  

---

## 👨‍💻 About Me

**Ajmal M**  
IIT Madras | Passionate about AI x Ethics x Impact_  
📧 [Email](mailto:ajmal@iitm.ac.in) | 🌐 [GitHub](https://github.com/AjmalMIITM)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more.
