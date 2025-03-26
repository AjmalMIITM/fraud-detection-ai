from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import shap
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename='api.log',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load the trained model
try:
    with open('rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

# Print the number of classes for debugging (also log it)
logging.info(f"Number of classes in the model: {rf_model.n_classes_}")

# Initialize SHAP explainer
try:
    explainer = shap.TreeExplainer(rf_model)
    logging.info("SHAP explainer initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize SHAP explainer: {str(e)}")
    raise

# Best threshold from Step 2
best_threshold = 0.03

app = Flask(__name__)

def validate_features(features):
    """Validate the input features."""
    if not isinstance(features, list):
        return False, "Features must be a list"
    if len(features) != 29:
        return False, f"Expected 29 features (V1-V28, Amount), got {len(features)}"
    if not all(isinstance(x, (int, float)) for x in features):
        return False, "All features must be numeric"
    return True, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.json
        if not data or 'features' not in data:
            logging.warning("Invalid input: 'features' key missing")
            return jsonify({'error': "Missing 'features' key in request body"}), 400

        features = data['features']
        is_valid, error_msg = validate_features(features)
        if not is_valid:
            logging.warning(f"Invalid input: {error_msg}")
            return jsonify({'error': error_msg}), 400

        logging.info(f"Received valid request with features: {features}")

        # Convert to DataFrame
        features_array = np.array([features])
        df = pd.DataFrame(features_array, columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])

        # Make prediction
        proba = rf_model.predict_proba(df)[:, 1]
        pred = (proba >= best_threshold).astype(int)[0]

        # Generate SHAP explanation
        shap_values = explainer.shap_values(df)
        
        # Log the structure of shap_values
        logging.info(f"Structure of shap_values: {type(shap_values)}")
        if isinstance(shap_values, list):
            logging.info(f"Length of shap_values list: {len(shap_values)}")
            for i, val in enumerate(shap_values):
                logging.info(f"shap_values[{i}] shape: {np.array(val).shape}")
        else:
            logging.info(f"shap_values shape: {shap_values.shape}")

        # Handle SHAP values based on their structure
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals = shap_values[1][0]  # SHAP values for Class=1 (fraud), first sample
        elif isinstance(shap_values, list) and len(shap_values) == 1:
            shap_vals = shap_values[0][0]  # First sample
        else:
            shap_vals = shap_values[0, :, 1]  # Shape: (29,)

        top_features_idx = np.argsort(np.abs(shap_vals))[-3:]  # Top 3 features
        top_features = df.columns[top_features_idx].tolist()
        top_values = df.iloc[0][top_features].values.tolist()
        feature_contributions = shap_vals[top_features_idx].tolist()

        # Dynamic explanation based on SHAP contributions
        explanation_parts = []
        for feature, value, contribution in zip(top_features, top_values, feature_contributions):
            if contribution > 0:
                explanation_parts.append(
                    f"{feature} (value: {value:.2f}, SHAP: {contribution:.4f}) increased the likelihood of fraud"
                )
            else:
                explanation_parts.append(
                    f"{feature} (value: {value:.2f}, SHAP: {contribution:.4f}) decreased the likelihood of fraud"
                )
        explanation = "Prediction influenced by: " + "; ".join(explanation_parts) + "."

        logging.info(f"Prediction: {pred}, Probability: {proba[0]}, Explanation: {explanation}")

        return jsonify({
            'fraud': int(pred),
            'probability': float(proba[0]),
            'explanation': explanation
        })

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)