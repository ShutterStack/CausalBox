# routers/preprocess_routes.py
from flask import Blueprint, request, jsonify
import pandas as pd
from utils.preprocessor import DataPreprocessor
import logging

preprocess_bp = Blueprint('preprocess', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

preprocessor = DataPreprocessor()

@preprocess_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload and preprocess a CSV file.
    Returns preprocessed DataFrame columns and data as JSON.
    Optional limit_rows to reduce response size for testing.
    """
    if 'file' not in request.files:
        return jsonify({"detail": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"detail": "Only CSV files are supported"}), 400

    limit_rows = request.args.get('limit_rows', type=int)

    try:
        logger.info(f"Received file: {file.filename}")
        df = pd.read_csv(file)
        logger.info(f"CSV read successfully, shape: {df.shape}")

        processed_df = preprocessor.preprocess(df)
        if limit_rows:
            processed_df = processed_df.head(limit_rows)
            logger.info(f"Limited to {limit_rows} rows.")

        response = {
            "columns": list(processed_df.columns),
            "data": processed_df.to_dict(orient="records")
        }
        logger.info(f"Preprocessed {len(response['data'])} records.")
        return jsonify(response)
    except pd.errors.EmptyDataError:
        logger.error("Empty CSV file uploaded.")
        return jsonify({"detail": "Empty CSV file"}), 400
    except pd.errors.ParserError:
        logger.error("Invalid CSV format.")
        return jsonify({"detail": "Invalid CSV format"}), 400
    except Exception as e:
        logger.exception(f"Unexpected error during file processing: {str(e)}")
        return jsonify({"detail": f"Failed to process file: {str(e)}"}), 500