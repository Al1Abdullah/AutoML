import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from rag.memory import store_dataset, get_dataset, store_model_results, get_model_results
from rag.rag_query import query_dataset_with_groq
from models.supervised import train_model as train_supervised
from models.unsupervised import train_unsupervised
from visuals.charts import (
    plot_histogram, plot_bar, plot_scatter, plot_box, plot_pie, plot_heatmap,
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_elbow_curve, plot_cluster_plot, plot_dendrogram, plot_tsne
)
import os
import logging
import json
import re
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, static_folder='frontend')
dataset_name = "active_dataset"

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    """Serve static files from the frontend directory."""
    return send_from_directory(app.static_folder, path)

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Handle CSV file uploads, store the dataset, and return a success message."""
    if 'file' not in request.files:
        logging.warning("No file part in upload request.")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.warning("No selected file in upload request.")
        return jsonify({"error": "No selected file"}), 400
    try:
        df = pd.read_csv(file)
        store_dataset(dataset_name, df)
        logging.info(f"Uploaded {df.shape[0]} rows and {df.shape[1]} columns.")
        return jsonify({"message": f"Uploaded {df.shape[0]} rows and {df.shape[1]} columns."})
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/columns', methods=['GET'])
def get_columns():
    """Return a list of column names from the currently loaded dataset."""
    df = get_dataset(dataset_name)
    if df is not None:
        return jsonify({"columns": list(df.columns)})
    logging.info("No dataset loaded when requesting columns.")
    return jsonify({"columns": []})

@app.route('/api/learning_type', methods=['GET'])
def get_learning_type():
    """Determine and return the learning type (supervised/unsupervised) and target column using LLM intelligence."""
    df = get_dataset(dataset_name)
    if df is None:
        logging.warning("No dataset uploaded when requesting learning type.")
        return jsonify({"error": "No dataset uploaded yet."}), 400

    dtypes_str = df.dtypes.to_string()
    prompt = (
        "You are an expert data scientist. Your task is to analyze a dataset and determine its learning type (supervised or unsupervised). "
        "If it's a supervised learning problem, you MUST identify the single target column that the other columns would predict. "
        "A target column is typically a label, outcome, or value that is being predicted (e.g., 'price', 'churn', 'diagnosis', 'category', 'sales'). "
        "If no such clear target column exists, it's an unsupervised problem. "
        "Respond ONLY with a JSON object, and nothing else. Do NOT include any introductory/concluding remarks, explanations, or markdown outside the JSON. "
        "The JSON must strictly follow this format: "
        "{\"learning_type\": \"Supervised\", \"target_column\": \"your_target_column_name\"} "
        "OR "
        "{\"learning_type\": \"Unsupervised\", \"target_column\": null}. "
        "\n\n"
        f"COLUMNS AND DATA TYPES:\n{dtypes_str}\n\n"
        f"DATA SAMPLE:\n{df.head().to_string()}"
    )

    try:
        response_text = query_dataset_with_groq(dataset_name, prompt).strip()
        logging.info(f"Raw LLM response for learning type: {response_text}")

        # Attempt to parse the JSON response
        try:
            data = json.loads(response_text)
            learning_type = data.get("learning_type", "Unsupervised")
            target_column = data.get("target_column")
            logging.info(f"Parsed LLM response - learning_type: {learning_type}, target_column: {target_column}")

            # Validate the target column if learning_type is Supervised
            if learning_type == "Supervised":
                if target_column is None or target_column not in df.columns:
                    logging.warning(f"LLM suggested supervised learning but target column '{target_column}' is invalid or not found. Defaulting to Unsupervised.")
                    learning_type = "Unsupervised"
                    target_column = None
            else:
                # If LLM says unsupervised, ensure target_column is null
                target_column = None

        except json.JSONDecodeError:
            logging.error(f"LLM response is not a valid JSON: {response_text}. Attempting regex fallback.")
            # Fallback: Try to extract using regex if JSON parsing fails (less reliable)
            match = re.search(r'"learning_type"\s*:\s*"(Supervised|Unsupervised)"(?:,\s*"target_column"\s*:\s*"?([a-zA-Z0-9_]+)?"?)?', response_text)
            if match:
                learning_type = match.group(1)
                target_column = match.group(2) if match.group(2) else None
                logging.info(f"Regex fallback parsed - learning_type: {learning_type}, target_column: {target_column}")

                if learning_type == "Supervised" and (target_column is None or target_column not in df.columns):
                    logging.warning(f"Regex fallback: Invalid target column '{target_column}' for supervised. Defaulting to Unsupervised.")
                    learning_type = "Unsupervised"
                    target_column = None
                elif learning_type == "Unsupervised":
                    target_column = None
            else:
                logging.error("Could not parse LLM response for learning type using regex fallback. Defaulting to Unsupervised.")
                learning_type = "Unsupervised"
                target_column = None

        return jsonify({"learning_type": learning_type, "target_column": target_column})

    except Exception as e:
        logging.error(f"An unexpected error occurred while determining learning type: {str(e)}", exc_info=True)
        # Fallback to a default in case of any error during Groq call or initial processing
        return jsonify({"learning_type": "Unsupervised", "target_column": None})

@app.route('/api/train', methods=['POST'])
def train_model_api():
    """Handle model training requests for both supervised and unsupervised learning."""
    data = request.json
    model_name = data.get('model_name')
    target_col = data.get('target_col')
    learning_type = data.get('learning_type')

    df = get_dataset(dataset_name)
    if df is None:
        logging.warning("No dataset uploaded when requesting model training.")
        return jsonify({"error": "No dataset uploaded yet."}), 400

    if learning_type == "Supervised":
        if not target_col or target_col == 'None':
            logging.warning("No target column provided for supervised training.")
            return jsonify({"error": "Please select a target column for supervised learning."}), 400
        
        model, metrics, y_test, y_pred, y_pred_proba, X_test = train_supervised(df, target_col, model_name)
        if model:
            store_model_results(dataset_name, model, y_test, y_pred, y_pred_proba, X_test)
            logging.info(f"{model_name} trained successfully for supervised learning.")
            return jsonify({"message": f"{model_name} trained successfully.", "metrics": metrics})
        else:
            logging.error(f"Failed to train {model_name} for supervised learning. Reason: {metrics}")
            return jsonify({"error": f"Failed to train {model_name}. Reason: {metrics}"}), 500
    else: # Unsupervised
        model, result = train_unsupervised(df, model_name)
        if model:
            logging.info(f"{model_name} trained successfully for unsupervised learning.")
            return jsonify({"message": f"{model_name} trained successfully.", "result": result.tolist() if hasattr(result, 'tolist') else result})
        else:
            logging.error(f"Failed to train {model_name} for unsupervised learning. Reason: {result}")
            return jsonify({"error": f"Failed to train {model_name}. Reason: {result}"}), 500

@app.route('/api/plot', methods=['POST'])
def generate_plot_api():
    """Generate and return a plot based on the requested type and columns."""
    data = request.json
    plot_type = data.get('plot_type')
    col1 = data.get('col1')
    col2 = data.get('col2')

    df = get_dataset(dataset_name)
    if df is None:
        logging.warning("No dataset loaded when requesting plot generation.")
        return jsonify({"error": "No data loaded."}), 400

    plot_functions = {
        "Histogram": plot_histogram,
        "Bar": plot_bar,
        "Scatter": plot_scatter,
        "Box": plot_box,
        "Pie": plot_pie,
        "Heatmap": plot_heatmap,
        "Elbow Curve": plot_elbow_curve,
        "Cluster Plot": plot_cluster_plot,
        "Dendrogram": plot_dendrogram,
        "t-SNE": plot_tsne,
        "Confusion Matrix": plot_confusion_matrix,
        "ROC Curve": plot_roc_curve,
        "Feature Importance Plot": plot_feature_importance
    }

    if plot_type not in plot_functions:
        logging.warning(f"Unsupported plot type requested: {plot_type}")
        return jsonify({"error": "Plot not supported."}), 400

    fig, err = None, None
    try:
        if plot_type == "Scatter":
            fig, err = plot_functions[plot_type](df, col1, col2, data.get('color_col'))
        elif plot_type == "Box":
            fig, err = plot_functions[plot_type](df, col1, col2)
        elif plot_type == "Heatmap":
            fig, err = plot_functions[plot_type](df)
        elif plot_type == "Elbow Curve":
            from utils.data_cleaner import prepare_data
            X_prepared, _ = prepare_data(df)
            fig, err = plot_functions[plot_type](X_prepared)
        elif plot_type == "Cluster Plot":
            from utils.data_cleaner import prepare_data
            from sklearn.cluster import KMeans
            X_prepared, _ = prepare_data(df)
            if X_prepared.empty:
                return jsonify({"error": "Data is empty after cleaning for Cluster Plot."}), 400
            # Perform KMeans clustering (e.g., with 3 clusters)
            n_clusters = 3 # Default number of clusters
            if len(X_prepared) < n_clusters:
                n_clusters = len(X_prepared) # Adjust n_clusters if data points are fewer
            if n_clusters == 0:
                return jsonify({"error": "Not enough data points to form clusters."}), 400
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X_prepared)
            labels = kmeans.labels_
            fig, err = plot_functions[plot_type](X_prepared, labels=labels)
        elif plot_type == "Dendrogram":
            from utils.data_cleaner import prepare_data
            X_prepared, _ = prepare_data(df)
            fig, err = plot_functions[plot_type](X_prepared)
        elif plot_type == "t-SNE":
            from utils.data_cleaner import prepare_data
            X_prepared, _ = prepare_data(df)
            fig, err = plot_functions[plot_type](X_prepared)
        elif plot_type in ["Confusion Matrix", "ROC Curve", "Feature Importance Plot"]:
            model_results = get_model_results(dataset_name)
            if not model_results:
                logging.warning(f"No trained model found for {plot_type} plot.")
                return jsonify({"error": "No trained model found. Please train a supervised model first."}), 400
            
            model = model_results['model']
            y_test = model_results['y_test']
            y_pred = model_results['y_pred']
            y_pred_proba = model_results['y_pred_proba']
            X_test = model_results['X_test']

            if plot_type == "Confusion Matrix":
                # Need to get class names. For simplicity, using unique values from y_test.
                class_names = [str(c) for c in sorted(pd.Series(y_test).unique())]
                fig, err = plot_functions[plot_type](y_test, y_pred, class_names)
            elif plot_type == "ROC Curve":
                if y_pred_proba is None:
                    logging.warning("ROC Curve requested but model does not provide probability predictions.")
                    return jsonify({"error": "ROC Curve requires probability predictions, which this model does not provide."}), 400
                fig, err = plot_functions[plot_type](y_test, y_pred_proba)
            elif plot_type == "Feature Importance Plot":
                if not hasattr(model, 'feature_importances_'):
                    logging.warning("Feature Importance Plot requested but model does not have feature importances.")
                    return jsonify({"error": "Model does not have feature importances to plot."}), 400
                # Feature names are from X_test columns
                feature_names = X_test.columns.tolist()
                fig, err = plot_functions[plot_type](model, feature_names)
        else:
            # Default case for plots that only need one column (e.g., Histogram, Bar, Pie)
            fig, err = plot_functions[plot_type](df, col1)

        if err:
            logging.error(f"Plot generation error for {plot_type}: {err}")
            return jsonify({"error": err}), 400
        
        # Save plot to a BytesIO object and encode to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return jsonify({'image': img_str})
    except Exception as e:
        logging.error(f"An unexpected error occurred during plot generation for {plot_type}: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

@app.route('/api/plot_options', methods=['GET'])
def plot_options():
    """Return a list of available plot options based on the dataset's learning type."""
    df = get_dataset(dataset_name)
    if df is None:
        logging.warning("No dataset uploaded when requesting plot options.")
        return jsonify({"error": "No dataset uploaded yet."}), 400

    # Get learning type from the dedicated endpoint
    learning_type_response = get_learning_type()
    learning_type_data = learning_type_response.get_json()
    learning_type = learning_type_data.get('learning_type', 'Unsupervised')

    if learning_type == "Supervised":
        plots = ["Histogram", "Bar", "Scatter", "Box", "Pie", "Heatmap", "Confusion Matrix", "ROC Curve", "Feature Importance Plot"]
    else:
        plots = ["Histogram", "Bar", "Scatter", "Box", "Pie", "Heatmap", "Cluster Plot", "Elbow Curve", "Dendrogram", "t-SNE"]

    # Ensure Scatter Plot is always available if there are at least two numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) >= 2 and "Scatter" not in plots:
        plots.insert(2, "Scatter") # Insert at a reasonable position
    
    return jsonify({"plots": plots})

@app.route('/api/ask', methods=['POST'])
def ask_question_api():
    """Handle user questions to the AI about the dataset."""
    data = request.json
    user_query = data.get('user_query')
    if not user_query:
        logging.warning("Empty user query received for AI assistant.")
        return jsonify({"error": "Please ask a question."}), 400
    
    answer = query_dataset_with_groq(dataset_name, user_query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(debug=True, port=5001)
