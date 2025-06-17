# routers/treatment_routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from utils.treatment_effects import TreatmentEffectAlgorithms
import logging

treatment_bp = Blueprint('treatment', __name__)
logger = logging.getLogger(__name__)

treatment_effect_algorithms = TreatmentEffectAlgorithms()

@treatment_bp.route('/estimate_ate', methods=['POST'])
def estimate_ate():
    """
    Estimate Average Treatment Effect (ATE) or Conditional Treatment Effect (CATE).
    Expects 'data' (list of dicts), 'treatment_col', 'outcome_col', 'covariates' (list of column names),
    and 'method' (string for estimation method).
    Returns ATE/CATE as float or dictionary.
    """
    try:
        payload = request.json
        if not payload or 'data' not in payload or 'treatment_col' not in payload or 'outcome_col' not in payload or 'covariates' not in payload:
            return jsonify({"detail": "Missing required ATE estimation parameters."}), 400

        df = pd.DataFrame(payload["data"])
        treatment_col = payload["treatment_col"]
        outcome_col = payload["outcome_col"]
        covariates = payload["covariates"]
        method = payload.get("method", "linear_regression").lower() # Default to linear regression

        logger.info(f"ATE/CATE request: treatment={treatment_col}, outcome={outcome_col}, method={method}, data shape: {df.shape}")

        if not all(col in df.columns for col in [treatment_col, outcome_col] + covariates):
            return jsonify({"detail": "Invalid column names provided for ATE estimation."}), 400

        if method == "linear_regression":
            result = treatment_effect_algorithms.linear_regression_ate(df, treatment_col, outcome_col, covariates)
        elif method == "propensity_score_matching":
            result = treatment_effect_algorithms.propensity_score_matching(df, treatment_col, outcome_col, covariates) # Placeholder
        elif method == "inverse_propensity_weighting":
            result = treatment_effect_algorithms.inverse_propensity_weighting(df, treatment_col, outcome_col, covariates) # Placeholder
        elif method == "t_learner":
            result = treatment_effect_algorithms.t_learner(df, treatment_col, outcome_col, covariates) # Placeholder
        elif method == "s_learner":
            result = treatment_effect_algorithms.s_learner(df, treatment_col, outcome_col, covariates) # Placeholder
        else:
            return jsonify({"detail": f"Unsupported treatment effect estimation method: {method}"}), 400

        logger.info(f"Estimated ATE/CATE using {method}: {result}")
        return jsonify({"result": result})

    except Exception as e:
        logger.exception(f"Error in ATE/CATE estimation: {str(e)}")
        return jsonify({"detail": f"ATE/CATE estimation failed: {str(e)}"}), 500