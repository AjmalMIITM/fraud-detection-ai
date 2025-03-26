**Project Proposal: LLM-Powered Fraud Detection AI (Final Update)** 
 
**Objective:** Develop an AI system to detect credit card fraud using machine learning, with a focus on explainability to assist financial institutions in understanding model decisions. 
 
**Dataset:** Credit Card Fraud Detection Dataset 2023 from Kaggle, containing 284,807 transactions with 492 fraud cases, featuring 28 anonymized features (V1-V28) and an Amount column. 
 
**Methodology:** 
- Trained a Random Forest model on the dataset using scikit-learn 1.2.2, achieving high accuracy on a balanced subset of the data. 
- Implemented SHAP (version 0.46.0) for explainability, identifying key features (e.g., V3, V11, V17) contributing to fraud predictions. 
- Deployed the model as a Flask API (Flask 3.0.0) for real-time predictions, accessible via a POST request to the /predict endpoint. 
- Overcame challenges with dependency compatibility (e.g., SHAP compilation issues, pandas 2.0.3 indexing changes) and SHAP output handling by dynamically processing shap_values based on their structure. 
 
**Deliverables:** 
- A trained Random Forest model (rf_model.pkl). 
- A Flask API (app.py) for fraud detection, returning predictions, probabilities, and SHAP explanations. 
- A Jupyter notebook (fraud_detection.ipynb) documenting the data preprocessing, model training, and evaluation process. 
- A GitHub repository (https://github.com/AjmalMIITM/fraud-detection-ai) containing all code, the trained model, and documentation (README.md and proposal.md). 
 
**Timeline (Actual):** 
- Week 1: Data preprocessing and model training completed as planned. 
- Week 2: SHAP implementation and Flask API development faced delays due to dependency issues and SHAP output mismatches. 
- Week 3: Resolved issues, tested the API, and submitted the project via GitHub. 
 
**Challenges and Solutions:** 
- Dependency conflicts with SHAP and numba were resolved by pinning compatible versions (shap==0.46.0, numba==0.57.0). 
- SHAP output for binary classification varied across versions; adapted the code to handle both list and array outputs. 
- Pandas 2.0.3 indexing changes caused errors; fixed by ensuring 1D indexing for feature selection. 
