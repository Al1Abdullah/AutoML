from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier
from utils.metrics import classification_metrics, regression_metrics
from utils.data_cleaner import prepare_data
import pandas as pd
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(df, target_column, model_name):
    """Trains a supervised machine learning model based on the specified model name.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_column (str): The name of the target column.
        model_name (str): The name of the model to train (e.g., "Logistic Regression", "Random Forest").

    Returns:
        tuple: A tuple containing:
            - model: The trained model object.
            - metrics (dict): A dictionary of evaluation metrics.
            - y_test (pd.Series): Actual target values from the test set.
            - y_pred (np.array): Predicted target values for the test set.
            - y_pred_proba (np.array, optional): Predicted probabilities for classification tasks.
            - X_test (pd.DataFrame): Feature values from the test set.
            - error (str, optional): An error message if training fails.
    """
    try:
        # Prepare data: clean, encode, scale, and split into features (X) and target (y)
        X, y, label_encoders, is_classification = prepare_data(df, target_column)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = None
        # Initialize the selected model
        if model_name == "Logistic Regression":
            if not is_classification:
                return None, "Logistic Regression is for classification tasks.", None, None, None, None
            model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
        elif model_name == "Naive Bayes":
            if not is_classification:
                return None, "Naive Bayes is for classification tasks.", None, None, None, None
            model = GaussianNB()
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            if is_classification:
                model = RandomForestClassifier(random_state=42)
            else:
                model = RandomForestRegressor(random_state=42)
        elif model_name == "SVM":
            if is_classification:
                model = SVC(probability=True, random_state=42) # probability=True for ROC curve
            else:
                model = SVR()
        elif model_name == "KNN":
            if not is_classification:
                return None, "KNN is for classification tasks.", None, None, None, None
            model = KNeighborsClassifier()
        elif model_name == "XGBoost":
            if is_classification:
                model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
            else:
                model = XGBRegressor(random_state=42)
        elif model_name == "CatBoost":
            if not is_classification:
                return None, "CatBoost is for classification tasks.", None, None, None, None
            model = CatBoostClassifier(verbose=0, random_state=42) # verbose=0 to suppress output
        elif model_name == "Linear Regression":
            if is_classification:
                return None, "Linear Regression is for regression tasks.", None, None, None, None
            model = LinearRegression()
        else:
            return None, "Model not found.", None, None, None, None

        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = None

        # Get prediction probabilities for classification models (needed for ROC curve)
        if is_classification and hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

        # Calculate evaluation metrics
        if is_classification:
            metrics = classification_metrics(y_test, y_pred)
        else:
            metrics = regression_metrics(y_test, y_pred)

        logging.info(f"Successfully trained {model_name} model.")
        return model, metrics, y_test, y_pred, y_pred_proba, X_test
    except Exception as e:
        logging.error(f"An error occurred during model training for {model_name}: {e}", exc_info=True)
        return None, f"An error occurred during model training: {e}", None, None, None, None
