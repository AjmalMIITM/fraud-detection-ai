# 🛡️ LLM-Powered Fraud Detection with SHAP Explainability

This project presents a fraud detection system powered by a Large Language Model (LLM)-enhanced feature engineering pipeline and a Random Forest classifier. Built using the 2023 Kaggle Credit Card Fraud Dataset, it combines **real-time fraud detection** with **SHAP-based explainability**, deployed via a **Flask API**.

Designed for real-world FinTech applications in high-risk zones (e.g., UAE, India), the system ensures both **accuracy** and **interpretability**, essential for high-stakes financial environments.

---

## 📁 Project Structure

- `fraud_detection.ipynb` – Model training, evaluation, and SHAP-based analysis  
- `app.py` – Flask API backend for real-time predictions  
- `rf_model.pkl` – Pre-trained Random Forest model  
- `proposal.md` – Challenges, problem framing, and future work  
- `README.md` – Project documentation (you're reading it)

---

## ⚙️ Setup Instructions

### 1. Clone the repository:

```bash
git clone https://github.com/AjmalMIITM/fraud-detection-ai.git
cd fraud-detection-ai
```

### 2. Install dependencies:

```bash
pip install flask==3.0.0 pandas==2.0.3 numpy==1.24.4 \
scikit-learn==1.2.2 shap==0.46.0 scipy==1.9.3 numba==0.57.0
```

### 3. Run the Flask app:

```bash
python app.py
```

---

## 🔍 API Usage

You can test the API locally or via the deployed endpoint.

### ➤ Local Prediction

Start the server:

```bash
python app.py
```

Send a sample POST request:

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [-0.260648, -0.469648, 2.496266, -0.083724, 0.129681, 0.732898, 0.519014, -0.130006, 0.727159, 0.637735, -1.289146, 0.507876, 0.019821, 1.443803, 0.151603, -0.339666, -0.673666, -0.117375, 0.450852, -0.114963, -0.110552, 0.217606, -0.134794, 0.165959, 0.126280, -0.434824, -0.081230, -0.151045, 17982.10]}' \
http://127.0.0.1:5000/predict
```

### ➤ Cloud Prediction (Deployed)

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [-0.260648, -0.469648, 2.496266, -0.083724, 0.129681, 0.732898, 0.519014, -0.130006, 0.727159, 0.637735, -1.289146, 0.507876, 0.019821, 1.443803, 0.151603, -0.339666, -0.673666, -0.117375, 0.450852, -0.114963, -0.110552, 0.217606, -0.134794, 0.165959, 0.126280, -0.434824, -0.081230, -0.151045, 17982.10]}' \
https://fraud-detection-ai.onrender.com/predict
```

---

## 📊 SHAP Explainability

> SHAP (SHapley Additive exPlanations) is used to interpret the model's fraud predictions. It highlights how individual features contribute to each prediction, aiding transparency and trust in high-risk environments.

✅ Coming soon: SHAP summary plot visualization for top feature impacts.

---

## 🧠 Dataset Used

- **Name**: Credit Card Fraud Detection Dataset 2023  
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Properties**: Highly imbalanced dataset with anonymized numerical features representing transaction patterns.

---

## 📈 Model Overview

- **Classifier**: Random Forest  
- **Explainability**: SHAP values  
- **Deployment**: Flask API  
- **Performance**: Optimized for precision and recall under class imbalance

---

## 🛡️ Real-World Use Case

Designed for fraud detection pipelines in:

- Digital payment gateways (e.g., UPI, SWIFT)
- FinTech apps in regulatory-heavy zones (e.g., GCC, India)
- High-volume transaction auditing systems

---

## 📌 Future Enhancements

- 🔁 Integration with **BERT embeddings** for behavioral context  
- 📊 Gradio dashboard for non-technical users  
- ⚙️ Auto-retraining pipeline  
- 📄 Research publication targeting IEEE/ACM FinAI tracks

---

## 👨‍💻 Author

**Ajmal M**  
B.S. in Data Science, IIT Madras  
📧 [ajmal@iitm.ac.in](mailto:ajmal@iitm.ac.in)  
🔗 [GitHub](https://github.com/AjmalMIITM)

> 💬 Open to research collaborations and internships in AI for Finance, Compliance, and Smart City domains.

---
Answer from Perplexity: pplx.ai/share
