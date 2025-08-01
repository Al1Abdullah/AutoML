from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from utils.data_cleaner import prepare_data
import pandas as pd
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_unsupervised(df, model_name, n_clusters=3, eps=0.5, min_samples=5, n_components=2):
    """Trains an unsupervised machine learning model based on the specified model name.

    Args:
        df (pd.DataFrame): The input DataFrame for unsupervised learning.
        model_name (str): The name of the unsupervised model to train (e.g., "KMeans", "DBSCAN", "PCA").
        n_clusters (int, optional): Number of clusters for KMeans. Defaults to 3.
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other for DBSCAN. Defaults to 0.5.
        min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point for DBSCAN. Defaults to 5.
        n_components (int, optional): Number of components to keep for PCA. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - fitted_model: The trained unsupervised model object.
            - result: The clustering labels or transformed data.
            - error (str, optional): An error message if training fails.
    """
    try:
        # Prepare data for unsupervised learning (cleaning and scaling)
        df_prepared, _ = prepare_data(df)
        
        if df_prepared.empty:
            logging.warning("Prepared DataFrame is empty for unsupervised training.")
            return None, "Prepared data is empty."

        model = None
        # Initialize the selected unsupervised model
        if model_name == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif model_name == "DBSCAN":
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif model_name == "PCA":
            model = PCA(n_components=n_components)
        else:
            logging.warning(f"Unsupervised model not supported: {model_name}")
            return None, "Model not supported."

        # Fit the model to the prepared data
        fitted_model = model.fit(df_prepared)

        result = None
        # Extract results based on the model type
        if hasattr(fitted_model, 'labels_'):
            result = fitted_model.labels_
            logging.info(f"KMeans/DBSCAN trained. Clusters/labels generated.")
        elif hasattr(fitted_model, 'components_'):
            result = fitted_model.transform(df_prepared)
            logging.info(f"PCA trained. Data transformed to {n_components} components.")
        else:
            logging.info(f"Unsupervised model {model_name} trained, but no specific labels or components found.")

        return fitted_model, result
    except Exception as e:
        logging.error(f"An error occurred during unsupervised model training for {model_name}: {e}", exc_info=True)
        return None, f"An error occurred during model training: {e}"
