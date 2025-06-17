# routers/visualize_routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from utils.graph_utils import visualize_graph
import networkx as nx
import numpy as np
import logging

visualize_bp = Blueprint('visualize', __name__)
logger = logging.getLogger(__name__)

@visualize_bp.route('/graph', methods=['POST'])
def get_graph_visualization():
    """
    Generate a causal graph visualization from an adjacency matrix.
    Expects 'graph' (adjacency matrix as list of lists) and 'nodes' (list of node names).
    Returns Plotly JSON for the graph.
    """
    try:
        payload = request.json
        if not payload or 'graph' not in payload or 'nodes' not in payload:
            return jsonify({"detail": "Missing 'graph' or 'nodes' in request payload."}), 400

        adj_matrix = np.array(payload["graph"])
        nodes = payload["nodes"]

        logger.info(f"Received graph visualization request for {len(nodes)} nodes.")

        # Reconstruct networkx graph from adjacency matrix and node names
        graph_nx = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        
        # Map integer node labels back to original column names if necessary
        # Assuming nodes are ordered as they appear in the original dataframe or provided in 'nodes'
        mapping = {i: node_name for i, node_name in enumerate(nodes)}
        graph_nx = nx.relabel_nodes(graph_nx, mapping)

        graph_json = visualize_graph(graph_nx)
        logger.info("Generated graph visualization JSON.")
        return jsonify({"graph": graph_json})

    except Exception as e:
        logger.exception(f"Error generating graph visualization: {str(e)}")
        return jsonify({"detail": f"Failed to generate visualization: {str(e)}"}), 500