# Fraud Detection AI 
 
This project implements an LLM-powered fraud detection system using a Random Forest model trained on the Credit Card Fraud Detection Dataset 2023 from Kaggle. The model is deployed as a Flask API, providing real-time fraud predictions with explainability through SHAP values. 
 
## Project Structure 
- `fraud_detection.ipynb`: Jupyter notebook with the model training and evaluation code. 
- `app.py`: Flask API for serving the fraud detection model. 
- `rf_model.pkl`: Pre-trained Random Forest model. 
- `README.md`: Project overview and instructions. 
- `proposal.md`: Updated project proposal with challenges and solutions. 
 
## Setup Instructions 
1. Clone the repository: 
   ``` 
   git clone https://github.com/AjmalMIITM/fraud-detection-ai.git 
   cd fraud-detection-ai 
   ``` 
2. Install dependencies: 
   ``` 
   pip install flask==3.0.0 pandas==2.0.3 numpy==1.24.4 scikit-learn==1.2.2 shap==0.46.0 scipy==1.9.3 numba==0.57.0 
   ``` 
3. Run the Flask app: 
   ``` 
   python app.py 
   ``` 
4. Test the API: 
   ``` 
   curl -X POST -H "Content-Type: application/json" -d "{\"features\": [-0.260648, -0.469648, 2.496266, -0.083724, 0.129681, 0.732898, 0.519014, -0.130006, 0.727159, 0.637735, -1.289146, 0.507876, 0.019821, 1.443803, 0.151603, -0.339666, -0.673666, -0.117375, 0.450852, -0.114963, -0.110552, 0.217606, -0.134794, 0.165959, 0.126280, -0.434824, -0.081230, -0.151045, 17982.10]}" http://localhost:5000/predict 
   ``` 
 
## Dataset 
The model was trained on the [Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle. 
 
## Author 
Ajmal 
