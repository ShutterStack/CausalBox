# utils/preprocessor.py
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
    
def summarize_dataframe_for_chatbot(data_list):
        """
        Generates a test summary of the DataFrame for chatbot interaction."""
        if not data_list:
            return "No data loaded."
        df = pd.DataFrame(data_list)
        nums_rows, num_cols = df.shape

        col_info = []
        for col in df.columns:
            dtype = df[col].dtype
            unique_vals = df[col].nunique()
            missing_count = df[col].isnull().sum()

            info = f"-{col} (Type:{dtype}"
            if pd.api.types.is_numeric_dtype(df[col]):
                info +=f", Min:{df[col].min():.2f}, Max:{df[col].max():.2f}"
            else:
                info += f", Unique:{unique_vals}"
            
            if missing_count > 0:
                info += f", Missing:{missing_count}"
            info += ")"
            col_info.append(info)
        summary = (f"Dataset Summary:\n- Rows: {nums_rows}, Columns: {num_cols}\nColumns:\n" + "\n".join(col_info))
        return summary