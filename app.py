from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import shap

# Load the trained model
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Print the number of classes for debugging
print(f"Number of classes in the model: {rf_model.n_classes_}")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(rf_model)

# Best threshold from Step 2
best_threshold = 0.03

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        features = np.array([data['features']])  # Expecting [V1-V28, Amount]
        df = pd.DataFrame(features, columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
        
        # Make prediction
        proba = rf_model.predict_proba(df)[:, 1]
        pred = (proba >= best_threshold).astype(int)[0]
        
        # Generate SHAP explanation
        shap_values = explainer.shap_values(df)
        
        # Debug: Print the structure of shap_values
        print("Structure of shap_values:", type(shap_values))
        if isinstance(shap_values, list):
            print(f"Length of shap_values list: {len(shap_values)}")
            for i, val in enumerate(shap_values):
                print(f"shap_values[{i}] shape: {np.array(val).shape}")
        else:
            print(f"shap_values shape: {shap_values.shape}")
        
        # Handle SHAP values based on their structure
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Binary classification: shap_values is [shap_values_class_0, shap_values_class_1]
            shap_vals = shap_values[1][0]  # SHAP values for Class=1 (fraud), first sample
        elif isinstance(shap_values, list) and len(shap_values) == 1:
            # SHAP returned a list with one element (likely for the positive class)
            shap_vals = shap_values[0][0]  # First sample
        else:
            # SHAP returned a single array of shape (n_samples, n_features, n_classes)
            # Extract SHAP values for Class=1 (fraud), first sample
            shap_vals = shap_values[0, :, 1]  # Shape: (29,)
        
        top_features_idx = np.argsort(np.abs(shap_vals))[-3:]  # Top 3 features, shape: (3,)
        top_features = df.columns[top_features_idx].tolist()  # Convert to list for JSON serialization
        top_values = df.iloc[0][top_features].values.tolist()  # Convert to list for JSON serialization
        feature_contributions = shap_vals[top_features_idx].tolist()  # Convert to list for JSON serialization
        
        explanation = f"Flagged due to significant contributions from {top_features} (values: {top_values}, SHAP contributions: {feature_contributions}). These features often indicate unusual patterns, such as high transaction amounts or suspicious activity."
        
        return jsonify({
            'fraud': int(pred),
            'probability': float(proba[0]),
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)