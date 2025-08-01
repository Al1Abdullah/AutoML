"""Module for calculating and returning various machine learning evaluation metrics."""

from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, mean_squared_error, precision_score, recall_score, f1_score
import numpy as np
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def classification_metrics(y_true, y_pred):
    """Calculates common classification metrics.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1-score, and confusion matrix.
    """
    logging.info("Calculating classification metrics.")
    try:
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        logging.info("Classification metrics calculated successfully.")
        return metrics
    except Exception as e:
        logging.error(f"Error calculating classification metrics: {e}", exc_info=True)
        return {"error": f"Failed to calculate classification metrics: {e}"}

def regression_metrics(y_true, y_pred):
    """Calculates common regression metrics.

    Args:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        dict: A dictionary containing R2 score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
    """
    logging.info("Calculating regression metrics.")
    try:
        metrics = {
            "R2 Score": r2_score(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
        }
        logging.info("Regression metrics calculated successfully.")
        return metrics
    except Exception as e:
        logging.error(f"Error calculating regression metrics: {e}", exc_info=True)
        return {"error": f"Failed to calculate regression metrics: {e}"}
