# utils/prediction_models.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import numpy as np

def train_predict_random_forest(data_list, target_col, feature_cols, prediction_type='regression'):
    """
    Trains a Random Forest model and performs prediction/evaluation.

    Args:
        data_list (list of dict): List of dictionaries representing the dataset.
        target_col (str): Name of the target variable.
        feature_cols (list): List of names of feature variables.
        prediction_type (str): 'regression' or 'classification'.

    Returns:
        dict: A dictionary containing model results (metrics, predictions, feature importances).
    """
    df = pd.DataFrame(data_list)
    
    if not all(col in df.columns for col in feature_cols + [target_col]):
        missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
        raise ValueError(f"Missing columns in data: {missing_cols}")

    X = df[feature_cols]
    y = df[target_col]

    # Handle categorical features if any
    X = pd.get_dummies(X, drop_first=True) # One-hot encode categorical features

    # Split data for robust evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    if prediction_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results['model_type'] = 'Regression'
        results['r2_score'] = r2_score(y_test, y_pred)
        results['mean_squared_error'] = mean_squared_error(y_test, y_pred)
        results['root_mean_squared_error'] = np.sqrt(mean_squared_error(y_test, y_pred))
        results['actual_vs_predicted'] = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_dict(orient='list')

    elif prediction_type == 'classification':
        # Ensure target variable is suitable for classification (e.g., integer/categorical)
        # You might need more robust handling for different target types here
        if y.dtype == 'object' or y.dtype.name == 'category':
            y_train = y_train.astype('category').cat.codes
            y_test = y_test.astype('category').cat.codes
            y_unique_labels = df[target_col].astype('category').cat.categories.tolist()
            results['class_labels'] = y_unique_labels
        else:
            y_unique_labels = sorted(y.unique().tolist())
            results['class_labels'] = y_unique_labels


        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results['model_type'] = 'Classification'
        results['accuracy'] = accuracy_score(y_test, y_pred)
        
        # Precision, Recall, F1-score - use 'weighted' average for multi-class
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
        results['precision'] = precision
        results['recall'] = recall
        results['f1_score'] = f1
        
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
    else:
        raise ValueError("prediction_type must be 'regression' or 'classification'")

    # Feature Importance (common for both)
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        results['feature_importances'] = feature_importances.to_dict()

    return results