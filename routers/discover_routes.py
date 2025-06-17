# routers/discover_routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from utils.casual_algorithms import CausalDiscoveryAlgorithms
import logging

discover_bp = Blueprint('discover', __name__)
logger = logging.getLogger(__name__)

causal_discovery_algorithms = CausalDiscoveryAlgorithms()

@discover_bp.route('/', methods=['POST'])
def discover_causal_graph():
    """
    Discover causal graph from input data using selected algorithm.
    Expects 'data' key with list of dicts (preprocessed DataFrame records) and 'algorithm' string.
    Returns graph as adjacency matrix.
    """
    try:
        payload = request.json
        if not payload or 'data' not in payload:
            return jsonify({"detail": "Invalid request payload: 'data' key missing."}), 400

        df = pd.DataFrame(payload["data"])
        algorithm = payload.get("algorithm", "pc").lower() # Default to PC

        logger.info(f"Received discovery request with algorithm: {algorithm}, data shape: {df.shape}")

        if algorithm == "pc":
            adj_matrix = causal_discovery_algorithms.pc_algorithm(df)
        elif algorithm == "ges":
            adj_matrix = causal_discovery_algorithms.ges_algorithm(df) # Placeholder
        elif algorithm == "notears":
            adj_matrix = causal_discovery_algorithms.notears_algorithm(df) # Placeholder
        else:
            return jsonify({"detail": f"Unsupported causal discovery algorithm: {algorithm}"}), 400

        logger.info(f"Causal graph discovered using {algorithm}.")
        return jsonify({"graph": adj_matrix.tolist()})

    except Exception as e:
        logger.exception(f"Error in causal discovery: {str(e)}")
        return jsonify({"detail": f"Causal discovery failed: {str(e)}"}), 500