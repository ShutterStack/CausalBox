# routers/intervene_routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from utils.do_calculus import DoCalculus # Will be used for more advanced intervention
import networkx as nx # Assuming graph is passed or re-discovered
import logging

intervene_bp = Blueprint('intervene', __name__)
logger = logging.getLogger(__name__)

@intervene_bp.route('/', methods=['POST'])
def perform_intervention():
    """
    Perform causal intervention on data.
    Expects 'data' (list of dicts), 'intervention_var' (column name),
    'intervention_value' (numeric), and optionally 'graph' (adjacency matrix).
    Returns intervened data as list of dicts.
    """
    try:
        payload = request.json
        if not payload or 'data' not in payload or 'intervention_var' not in payload or 'intervention_value' not in payload:
            return jsonify({"detail": "Missing required intervention parameters."}), 400

        df = pd.DataFrame(payload["data"])
        intervention_var = payload["intervention_var"]
        intervention_value = payload["intervention_value"]
        graph_adj_matrix = payload.get("graph") # Optional: pass pre-discovered graph

        logger.info(f"Intervention request: var={intervention_var}, value={intervention_value}, data shape: {df.shape}")

        if intervention_var not in df.columns:
            return jsonify({"detail": f"Intervention variable '{intervention_var}' not found in data"}), 400

        # For a more advanced do-calculus, you'd need the graph structure.
        # Here, a simplified direct intervention is applied first.
        # If graph_adj_matrix is provided, you could convert it to networkx.
        # For full do-calculus, the DoCalculus class would need a proper graph.
        
        df_intervened = df.copy()
        df_intervened[intervention_var] = intervention_value
        
        # Placeholder for propagating effects using a graph if provided
        # if graph_adj_matrix:
        #    graph_nx = nx.from_numpy_array(np.array(graph_adj_matrix), create_using=nx.DiGraph)
        #    do_calculus_engine = DoCalculus(graph_nx)
        #    df_intervened = do_calculus_engine.intervene(df_intervened, intervention_var, intervention_value)
        #    logger.info("Propagated effects using do-calculus (simplified).")

        logger.info(f"Intervened data shape: {df_intervened.shape}")
        return jsonify({"intervened_data": df_intervened.to_dict(orient="records")})

    except Exception as e:
        logger.exception(f"Error in intervention: {str(e)}")
        return jsonify({"detail": f"Intervention failed: {str(e)}"}), 500