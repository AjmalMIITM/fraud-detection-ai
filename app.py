from flask import Flask, request, jsonify
import pickle
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the model and SHAP explainer
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
explainer = shap.TreeExplainer(rf_model)

@app.route('/')
def home():
    return "Welcome to the Fraud Detection AI API! Use the /predict endpoint to make predictions. Example: POST to /predict with {'features': [-0.260648, -0.469648, 2.496266, -0.083724, 0.129681, 0.732898, 0.519014, -0.130006, 0.727159, 0.637735, -1.289146, 0.507876, 0.019821, 1.443803, 0.151603, -0.339666, -0.673666, -0.117375, 0.450852, -0.114963, -0.110552, 0.217606, -0.134794, 0.165959, 0.126280, -0.434824, -0.081230, -0.151045, 17982.10]}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get('features')
    if not features or len(features) != 29:
        return jsonify({'error': 'Expected 29 features (V1-V28, Amount), got {}'.format(len(features) if features else 0)}), 400
    features = np.array(features).reshape(1, -1)
    prediction = rf_model.predict(features)[0]
    probability = rf_model.predict_proba(features)[0][1]
    shap_values = explainer.shap_values(features)
    top_features = np.argsort(np.abs(shap_values[1][0]))[-3:]
    explanation = "Prediction influenced by: "
    for i in top_features:
        explanation += f"V{i+1} (value: {features[0][i]:.2f}, SHAP: {shap_values[1][0][i]:.4f}) {'increased' if shap_values[1][0][i] > 0 else 'decreased'} the likelihood of fraud; "
    return jsonify({
        'fraud': int(prediction),
        'probability': float(probability),
        'explanation': explanation.strip('; ')
    })

if __name__ == '__main__':
    app.run(debug=True)