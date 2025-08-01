import pandas as pd
import os
import pickle
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CACHE_DIR = ".cache"
# Create the cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
    logging.info(f"Created cache directory: {CACHE_DIR}")

def get_dataset_path(name):
    """Constructs the file path for a dataset pickle file."""
    return os.path.join(CACHE_DIR, f"{name}.pkl")

def get_model_results_path(name):
    """Constructs the file path for model results pickle file."""
    return os.path.join(CACHE_DIR, f"{name}_model_results.pkl")

def store_dataset(name, dataframe):
    """Saves a DataFrame to a pickle file in the cache directory.

    Args:
        name (str): The name to associate with the dataset (used for filename).
        dataframe (pd.DataFrame): The DataFrame to be stored.
    """
    path = get_dataset_path(name)
    try:
        dataframe.to_pickle(path)
        logging.info(f"Dataset '{name}' stored successfully at {path}")
    except Exception as e:
        logging.error(f"Error storing dataset '{name}' to {path}: {e}", exc_info=True)
    
def get_dataset(name):
    """Loads a DataFrame from a pickle file in the cache directory.

    Args:
        name (str): The name of the dataset to retrieve.

    Returns:
        pd.DataFrame or None: The loaded DataFrame if found, otherwise None.
    """
    path = get_dataset_path(name)
    if os.path.exists(path):
        try:
            df = pd.read_pickle(path)
            logging.info(f"Dataset '{name}' loaded successfully from {path}")
            return df
        except Exception as e:
            logging.error(f"Error loading dataset '{name}' from {path}: {e}", exc_info=True)
            return None
    logging.info(f"Dataset '{name}' not found at {path}")
    return None

def store_model_results(name, model, y_test, y_pred, y_pred_proba, X_test):
    """Saves trained model, test data, predictions, and probabilities to a pickle file.

    Args:
        name (str): The name to associate with the model results.
        model: The trained model object.
        y_test (pd.Series): Actual target values from the test set.
        y_pred (np.array): Predicted target values for the test set.
        y_pred_proba (np.array, optional): Predicted probabilities for classification tasks.
        X_test (pd.DataFrame): Feature values from the test set.
    """
    path = get_model_results_path(name)
    results = {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "X_test": X_test
    }
    try:
        with open(path, 'wb') as f:
            pickle.dump(results, f)
        logging.info(f"Model results for '{name}' stored successfully at {path}")
    except Exception as e:
        logging.error(f"Error storing model results for '{name}' to {path}: {e}", exc_info=True)

def get_model_results(name):
    """Loads trained model, test data, predictions, and probabilities from a pickle file.

    Args:
        name (str): The name of the model results to retrieve.

    Returns:
        dict or None: A dictionary containing model results if found, otherwise None.
    """
    path = get_model_results_path(name)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                results = pickle.load(f)
            logging.info(f"Model results for '{name}' loaded successfully from {path}")
            return results
        except Exception as e:
            logging.error(f"Error loading model results for '{name}' from {path}: {e}", exc_info=True)
            return None
    logging.info(f"Model results for '{name}' not found at {path}")
    return None
