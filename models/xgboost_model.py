import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from utils.metrics import classification_metrics, regression_metrics

def encode_dataframe(df):
    label_encoders = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def train_xgboost(df, target_column, task='classification'):
    df = df.dropna()
    df, encoders = encode_dataframe(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if task == 'classification':
        model = XGBClassifier()
    else:
        model = XGBRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task == 'classification':
        metrics = classification_metrics(y_test, y_pred)
    else:
        metrics = regression_metrics(y_test, y_pred)

    return model, metrics
