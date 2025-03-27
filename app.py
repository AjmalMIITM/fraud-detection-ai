import os
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import shap

app = Flask(__name__)

# Load the model and SHAP explainer
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
explainer = shap.TreeExplainer(rf_model)

@app.route('/')
def home():
    return "Welcome to the Fraud Detection AI API! Use the /predict endpoint to make predictions..."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "This endpoint requires a POST request..."

    # Get and validate input
    data = request.get_json()
    features = data.get('features')
    if not features or len(features) != 29:
        return jsonify({'error': 'Expected 29 features (V1-V28, Amount), got {}'.format(len(features) if features else 0)}), 400

    # Validate numeric features
    try:
        features = [float(f) for f in features]
    except (ValueError, TypeError):
        return jsonify({'error': 'All features must be numeric'}), 400

    # Prepare features for prediction
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # Use probability with explicit threshold for prediction
    try:
        probability = rf_model.predict_proba(features_df)[0][1]
    except Exception as e:
        return jsonify({'error': f'Error computing prediction probability: {str(e)}'}), 500
    
    threshold = 0.5
    prediction = 1 if probability >= threshold else 0
    
    # Compute SHAP values for explainability
    try:
        shap_values = explainer.shap_values(features_df)
        # Debug: Log the shape of shap_values
        print(f"shap_values type: {type(shap_values)}, len: {len(shap_values) if isinstance(shap_values, list) else 'N/A'}")
        if isinstance(shap_values, list):
            print(f"shap_values[0] shape: {np.array(shap_values[0]).shape}")
            if len(shap_values) == 2:
                shap_values_for_class = shap_values[1]  # SHAP values for class 1 (fraud)
            elif len(shap_values) == 1 and len(np.array(shap_values[0]).shape) == 2 and np.array(shap_values[0]).shape[1] == 2:
                shap_values_for_class = shap_values[0][:, 1]
            else:
                shap_values_for_class = shap_values[0]
        else:
            # If shap_values is not a list, assume it's a numpy array
            shap_values_for_class = shap_values
        # Ensure shap_values_for_class is 2D
        shap_values_for_class = np.array(shap_values_for_class)
        if len(shap_values_for_class.shape) == 1:
            shap_values_for_class = shap_values_for_class.reshape(1, -1)
        print(f"shap_values_for_class shape: {shap_values_for_class.shape}")
    except Exception as e:
        return jsonify({'error': f'Error computing SHAP values: {str(e)}'}), 500

    # Generate explanation based on top 3 features
    try:
        shap_abs = np.abs(shap_values_for_class)
        print(f"shap_abs shape: {shap_abs.shape}")
        top_indices = np.argsort(shap_abs[0])[::-1]  # Sort indices in descending order
        top_features = [i for i in top_indices if shap_abs[0][i] > 0][:3]
        if len(top_features) > 0:
            explanation = "Prediction influenced by: "
            for i in top_features:
                explanation += f"{feature_names[i]} (value: {features[i]:.2f}, SHAP: {shap_values_for_class[0][i]:.4f}) {'increased' if shap_values_for_class[0][i] > 0 else 'decreased'} the likelihood of fraud; "
            explanation = explanation.strip('; ')
        else:
            explanation = "No significant features influenced the prediction (all SHAP values are zero)."
    except Exception as e:
        return jsonify({'error': f'Error generating SHAP explanation: {str(e)}'}), 500

    # Return response
    return jsonify({
        'fraud': int(prediction),
        'probability': float(probability),
        'explanation': explanation
    })
if __name__ == '__main__':
    app.run(debug=True)