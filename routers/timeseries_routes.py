# routers/timeseries_routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from utils.time_series_causal import perform_granger_causality

timeseries_bp = Blueprint('timeseries_bp', __name__)

@timeseries_bp.route('/discover_causality', methods=['POST'])
def discover_timeseries_causality():
    """
    API endpoint to perform time-series causal discovery (Granger Causality).
    """
    data = request.json.get('data')
    timestamp_col = request.json.get('timestamp_col')
    variables_to_analyze = request.json.get('variables_to_analyze')
    max_lags = request.json.get('max_lags', 1) # Default to 1 lag

    if not all([data, timestamp_col, variables_to_analyze]):
        return jsonify({"detail": "Missing required parameters for time-series causal discovery."}), 400

    if not isinstance(max_lags, int) or max_lags <= 0:
        return jsonify({"detail": "max_lags must be a positive integer."}), 400

    try:
        results = perform_granger_causality(data, timestamp_col, variables_to_analyze, max_lags)
        return jsonify({"results": results}), 200
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400
    except Exception as e:
        return jsonify({"detail": f"An error occurred during time-series causal discovery: {str(e)}"}), 500