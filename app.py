from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
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

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "This endpoint requires a POST request. Example: POST to /predict with {'features': [-0.260648, -0.469648, 2.496266, -0.083724, 0.129681, 0.732898, 0.519014, -0.130006, 0.727159, 0.637735, -1.289146, 0.507876, 0.019821, 1.443803, 0.151603, -0.339666, -0.673666, -0.117375, 0.450852, -0.114963, -0.110552, 0.217606, -0.134794, 0.165959, 0.126280, -0.434824, -0.081230, -0.151045, 17982.10]}"
    data = request.get_json()
    features = data.get('features')
    if not features or len(features) != 29:
        return jsonify({'error': 'Expected 29 features (V1-V28, Amount), got {}'.format(len(features) if features else 0)}), 400
    
    # Convert features to a DataFrame with proper feature names to suppress the warning
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    features_df = pd.DataFrame([features], columns=feature_names)
    features = np.array(features).reshape(1, -1)
    
    # Make predictions
    prediction = rf_model.predict(features_df)[0]
    probability = rf_model.predict_proba(features_df)[0][1]
    
    # Compute SHAP values
    shap_values = explainer.shap_values(features_df)
    
    # Debug: Print the shape of shap_values to understand its structure
    print("SHAP values structure:", [np.array(sv).shape for sv in shap_values])
    
    # Handle SHAP values based on the number of classes
    if len(shap_values) == 2:  # Binary classification: shap_values[0] for class 0, shap_values[1] for class 1
        shap_values_for_class = shap_values[1]  # Use SHAP values for class 1 (fraud)
    else:  # If only one set of SHAP values (e.g., single-class output), use the first set
        shap_values_for_class = shap_values[0]
    
    # Compute top features based on SHAP values (up to 3 features with non-zero SHAP values)
    shap_abs = np.abs(shap_values_for_class[0])
    top_indices = np.argsort(shap_abs)[::-1]  # Sort in descending order
    top_features = [i for i in top_indices if shap_abs[i] > 0][:3]  # Select up to 3 features with non-zero SHAP values
    
    # Build the explanation
    if len(top_features) > 0:
        explanation = "Prediction influenced by: "
        for i in top_features:
            explanation += f"{feature_names[i]} (value: {features[0][i]:.2f}, SHAP: {shap_values_for_class[0][i]:.4f}) {'increased' if shap_values_for_class[0][i] > 0 else 'decreased'} the likelihood of fraud; "
        explanation = explanation.strip('; ')
    else:
        explanation = "No significant features influenced the prediction (all SHAP values are zero)."
    
    return jsonify({
        'fraud': int(prediction),
        'probability': float(probability),
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(debug=True)