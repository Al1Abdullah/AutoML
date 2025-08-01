import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(df):
    """Cleans the input DataFrame by imputing missing values.

    - Numerical columns: Imputes missing values with the mean.
    - Categorical columns: Imputes missing values with the most frequent value.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    logging.info("Starting data cleaning process.")
    # Impute missing values for numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    if not numerical_cols.empty:
        logging.info(f"Imputing missing numerical values for columns: {list(numerical_cols)}")
        imputer_numerical = SimpleImputer(strategy='mean')
        df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    # Impute missing values for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        logging.info(f"Imputing missing categorical values for columns: {list(categorical_cols)}")
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
    
    logging.info("Data cleaning process completed.")
    return df

def prepare_data(df, target_column=None):
    """Prepares the DataFrame for machine learning by cleaning, encoding, and scaling.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str, optional): The name of the target column. If provided,
                                       data is prepared for supervised learning (X, y split).
                                       Otherwise, for unsupervised learning (all features).

    Returns:
        tuple: If target_column is provided:
                   (X (pd.DataFrame), y (pd.Series), label_encoders (dict), is_classification (bool))
               If target_column is None:
                   (df_prepared (pd.DataFrame), label_encoders (dict))
    """
    logging.info(f"Starting data preparation process. Target column: {target_column}")
    df = clean_data(df.copy()) # Ensure we work on a copy to avoid modifying original df
    label_encoders = {}
    is_classification = False

    # Encode categorical features (excluding the target column if it's categorical)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != target_column:
            logging.info(f"Encoding categorical feature: {col}")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    if target_column:
        # Supervised learning preparation
        logging.info(f"Preparing data for supervised learning with target: {target_column}")
        # Determine if it's a classification or regression task based on target column properties
        if df[target_column].dtype == 'object' or df[target_column].nunique() <= 10: # Heuristic for classification
            is_classification = True
            logging.info(f"Target column '{target_column}' identified as classification.")
            le = LabelEncoder()
            df[target_column] = le.fit_transform(df[target_column])
            label_encoders[target_column] = le
        else:
            logging.info(f"Target column '{target_column}' identified as regression.")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Scale numerical features in X
        numerical_cols = X.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            logging.info(f"Scaling numerical features in X: {list(numerical_cols)}")
            scaler = StandardScaler()
            X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
            
        logging.info("Data preparation for supervised learning completed.")
        return X, y, label_encoders, is_classification
    else:
        # Unsupervised learning preparation (scale all numerical features)
        logging.info("Preparing data for unsupervised learning.")
        numerical_cols = df.select_dtypes(include=['number']).columns
        if not numerical_cols.empty:
            logging.info(f"Scaling numerical features for unsupervised learning: {list(numerical_cols)}")
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logging.info("Data preparation for unsupervised learning completed.")
        return df.copy(), label_encoders
