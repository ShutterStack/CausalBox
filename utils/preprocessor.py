# utils/preprocessor.py
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess(self, df):
        """
        Preprocess DataFrame: handle missing values, encode categorical variables, scale numerical variables.
        """
        try:
            logger.info(f"Input DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            df_processed = df.copy()
            
            # Handle missing values
            logger.info("Handling missing values...")
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(df_processed[col]):
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                        logger.info(f"Filled numeric missing values in '{col}' with mean.")
                    else:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                        logger.info(f"Filled categorical missing values in '{col}' with mode.")
            
            # Encode categorical variables
            logger.info("Encoding categorical variables...")
            for col in df_processed.select_dtypes(include=['object', 'category']).columns:
                logger.info(f"Encoding column: {col}")
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            
            # Scale numerical variables
            logger.info("Scaling numerical variables...")
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Exclude columns that are now effectively categorical (post-label encoding)
                # This is a heuristic; ideally, identify original numeric columns.
                cols_to_scale = [col for col in numeric_cols if col not in self.label_encoders]
                if cols_to_scale:
                    df_processed[cols_to_scale] = self.scaler.fit_transform(df_processed[cols_to_scale])
                    logger.info(f"Scaled numeric columns: {cols_to_scale}")
            
            logger.info(f"Preprocessed DataFrame shape: {df_processed.shape}")
            return df_processed
        except Exception as e:
            logger.exception(f"Error preprocessing data: {str(e)}")
            raise