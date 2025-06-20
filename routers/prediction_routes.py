# routers/prediction_routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from utils.prediction_models import train_predict_random_forest

prediction_bp = Blueprint('prediction_bp', __name__)

@prediction_bp.route('/train_predict', methods=['POST'])
def train_predict():
    """
    API endpoint to train a Random Forest model and perform prediction/evaluation.
    """
    data = request.json.get('data')
    target_col = request.json.get('target_col')
    feature_cols = request.json.get('feature_cols')
    prediction_type = request.json.get('prediction_type')

    if not all([data, target_col, feature_cols, prediction_type]):
        return jsonify({"detail": "Missing required parameters for prediction."}), 400

    try:
        results = train_predict_random_forest(data, target_col, feature_cols, prediction_type)
        return jsonify({"results": results}), 200
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400
    except Exception as e:
        return jsonify({"detail": f"An error occurred during prediction: {str(e)}"}), 500