import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Consistent theme settings for plots
FIG_SIZE = (10, 6)
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 10
PRIMARY_COLOR = "#4C72B0"  # A nice blue
SECONDARY_COLOR = "#55A868" # A nice green

def plot_histogram(df, col):
    """Generates a histogram for a given numeric column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the numeric column to plot.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info(f"Generating histogram for column: {col}")
    if col not in df.columns:
        logging.error(f"Column '{col}' not found for histogram.")
        return None, f"Column '{col}' not found."
    if not pd.api.types.is_numeric_dtype(df[col]):
        logging.error(f"Column '{col}' is not numeric for histogram.")
        return None, "Histogram is only for numeric columns."
    
    plt.figure(figsize=FIG_SIZE)
    sns.set_style("whitegrid")
    
    # Calculate optimal bin width using Freedman-Diaconis rule
    try:
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        if iqr > 0:
            bin_width = 2 * iqr / (len(df[col]) ** (1/3))
            bins = int((df[col].max() - df[col].min()) / bin_width) if bin_width > 0 else 25
        else:
            bins = 25 # Default if IQR is zero
    except Exception as e:
        logging.warning(f"Could not calculate optimal bins for {col}: {e}. Using default 25 bins.")
        bins = 25
    
    ax = sns.histplot(df[col], kde=True, bins=bins, color=PRIMARY_COLOR, edgecolor='black', line_kws={'linewidth': 2, 'linestyle': '--'})
    
    # Add mean and median lines
    mean_val = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
    
    skewness = df[col].skew()
    plt.title(f'Distribution of {col} (Skewness: {skewness:.2f})', fontsize=TITLE_FONT_SIZE, weight='bold')
    plt.xlabel(col, fontsize=LABEL_FONT_SIZE)
    plt.ylabel('Density', fontsize=LABEL_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    logging.info(f"Histogram for {col} generated successfully.")
    return plt.gcf(), None

def plot_bar(df, col):
    """Generates a bar plot for a given categorical or discrete numeric column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to plot.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info(f"Generating bar plot for column: {col}")
    if col not in df.columns:
        logging.error(f"Column '{col}' not found for bar plot.")
        return None, f"Column '{col}' not found."
    
    plt.figure(figsize=FIG_SIZE)
    sns.set_style("whitegrid")
    
    counts = df[col].value_counts()
    # Handle too many categories by showing top N and grouping others
    if len(counts) > 15:
        logging.info(f"Column {col} has too many unique values ({len(counts)}). Showing top 14 and grouping others.")
        top_14 = counts.nlargest(14)
        other_sum = counts.nsmallest(len(counts) - 14).sum()
        top_14['Other'] = other_sum
        counts = top_14

    ax = sns.barplot(y=counts.index.astype(str), x=counts.values, palette="viridis", orient='h')
    
    # Add count labels to bars
    for i, v in enumerate(counts.values):
        ax.text(v + 1, i, str(v), color='black', va='center', fontsize=10)
        
    plt.title(f'Frequency of {col}', fontsize=TITLE_FONT_SIZE, weight='bold')
    plt.xlabel('Count', fontsize=LABEL_FONT_SIZE)
    plt.ylabel(col, fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    logging.info(f"Bar plot for {col} generated successfully.")
    return plt.gcf(), None

def plot_scatter(df, col1, col2, color_col=None):
    """Generates a scatter plot between two numeric columns, with optional coloring.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col1 (str): The name of the first numeric column (x-axis).
        col2 (str): The name of the second numeric column (y-axis).
        color_col (str, optional): The name of a column to use for coloring points.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info(f"Generating scatter plot for {col1} vs {col2}, colored by {color_col or 'None'}")
    if col1 not in df.columns or col2 not in df.columns:
        logging.error(f"One or both columns ({col1}, {col2}) not found for scatter plot.")
        return None, "One or both columns not found."
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        logging.error(f"Columns {col1} or {col2} are not numeric for scatter plot.")
        return None, "Scatter plots are only available for numeric columns."
    if color_col and color_col != 'None' and color_col not in df.columns:
        logging.error(f"Color column '{color_col}' not found for scatter plot.")
        return None, f"Color column '{color_col}' not found."
    
    try:
        plt.figure(figsize=FIG_SIZE)
        sns.set_style("whitegrid")
        hue = color_col if color_col and color_col != 'None' else None
        
        plot_df = df.dropna(subset=[col1, col2]) # Drop NaNs for plotting

        sns.scatterplot(data=plot_df, x=col1, y=col2, hue=hue, palette="coolwarm", s=50, alpha=0.6)
        
        # Add a linear regression trend line if both columns are numeric
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            # Ensure there's enough data for linear regression
            if len(plot_df) > 1:
                m, b, r_value, _, _ = stats.linregress(plot_df[col1], plot_df[col2])
                x_line = np.array([plot_df[col1].min(), plot_df[col1].max()])
                y_line = m * x_line + b
                plt.plot(x_line, y_line, color='red', linestyle='--', label=f'Trend Line (R² = {r_value**2:.2f})')
                plt.legend(fontsize=LEGEND_FONT_SIZE)
            else:
                logging.warning("Not enough data points for linear regression trend line.")

        plt.title(f'{col1} vs. {col2}', fontsize=TITLE_FONT_SIZE, weight='bold')
        plt.xlabel(col1, fontsize=LABEL_FONT_SIZE)
        plt.ylabel(col2, fontsize=LABEL_FONT_SIZE)
        plt.tight_layout()
        logging.info(f"Scatter plot for {col1} vs {col2} generated successfully.")
        return plt.gcf(), None
    except Exception as e:
        logging.error(f"An error occurred during scatter plot generation: {e}", exc_info=True)
        return None, f"An error occurred during plot generation: {e}"

def plot_box(df, continuous_var, group_var):
    """Generates a box plot to show the distribution of a continuous variable across categories of a grouping variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        continuous_var (str): The name of the continuous numeric column.
        group_var (str): The name of the categorical or discrete column for grouping.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info(f"Generating box plot for {continuous_var} by {group_var}")
    if continuous_var not in df.columns or group_var not in df.columns:
        logging.error(f"One or both columns ({continuous_var}, {group_var}) not found for box plot.")
        return None, "One or both columns not found."
    if not pd.api.types.is_numeric_dtype(df[continuous_var]):
        logging.error(f"Column '{continuous_var}' is not numeric for box plot.")
        return None, "Box plots require a numeric column for the x-axis."
    
    plt.figure(figsize=FIG_SIZE)
    sns.set_style("whitegrid")
    
    # Order categories by median of the continuous variable
    order = df.groupby(group_var)[continuous_var].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=continuous_var, y=group_var, palette="Set2", order=order, orient='h')
    
    plt.title(f'{continuous_var} by {group_var}', fontsize=TITLE_FONT_SIZE, weight='bold')
    plt.xlabel(continuous_var, fontsize=LABEL_FONT_SIZE)
    plt.ylabel(group_var, fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    logging.info(f"Box plot for {continuous_var} by {group_var} generated successfully.")
    return plt.gcf(), None

def plot_pie(df, col):
    """Generates a pie chart for a given categorical column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the categorical column to plot.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info(f"Generating pie chart for column: {col}")
    if col not in df.columns:
        logging.error(f"Column '{col}' not found for pie chart.")
        return None, f"Column '{col}' not found."
    
    counts = df[col].value_counts()
    # Handle too many categories by showing top N and grouping others
    if len(counts) > 7:
        logging.info(f"Column {col} has too many unique values ({len(counts)}). Showing top 6 and grouping others.")
        top_6 = counts.nlargest(6)
        other_sum = counts.nsmallest(len(counts) - 6).sum()
        top_6['Other'] = other_sum
        counts = top_6

    plt.figure(figsize=(8, 8)) # Pie charts often look better square
    
    explode = [0.03] * len(counts) # Slightly separate slices for better visual
    colors = sns.color_palette('pastel')[0:len(counts)]
    
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, explode=explode, colors=colors, pctdistance=0.85)
    centre_circle = plt.Circle((0,0),0.70,fc='white') # Donut chart effect
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title(f'Distribution of {col}', fontsize=TITLE_FONT_SIZE, weight='bold')
    plt.tight_layout()
    logging.info(f"Pie chart for {col} generated successfully.")
    return plt.gcf(), None

def plot_heatmap(df):
    """Generates a correlation heatmap for all numeric columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info("Generating correlation heatmap.")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        logging.error("Not enough numeric columns for a heatmap.")
        return None, "Not enough numeric columns for a heatmap."
    
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1, annot_kws={"size": 8})
    plt.title('Correlation Heatmap', fontsize=TITLE_FONT_SIZE, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=LABEL_FONT_SIZE)
    plt.yticks(rotation=0, fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    logging.info("Correlation heatmap generated successfully.")
    return plt.gcf(), None

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Generates a confusion matrix plot.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_names (list): List of class names for labels.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info("Generating confusion matrix.")
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix', fontsize=TITLE_FONT_SIZE, weight='bold')
        plt.ylabel('Actual', fontsize=LABEL_FONT_SIZE)
        plt.xlabel('Predicted', fontsize=LABEL_FONT_SIZE)
        plt.tight_layout()
        logging.info("Confusion matrix generated successfully.")
        return plt.gcf(), None
    except Exception as e:
        logging.error(f"Error generating confusion matrix: {e}", exc_info=True)
        return None, f"Error generating confusion matrix: {e}"

def plot_roc_curve(y_true, y_pred_proba, class_names=None):
    """Generates a Receiver Operating Characteristic (ROC) curve.

    Args:
        y_true (array-like): True binary labels.
        y_pred_proba (array-like): Target scores, probabilities of the positive class.
        class_names (list, optional): List of class names. Not directly used in plot but good for context.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info("Generating ROC curve.")
    try:
        # Handle multi-class or binary probability predictions
        if y_pred_proba.ndim == 1: # Binary classification, single probability array
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        elif y_pred_proba.shape[1] == 2: # Binary classification, two columns of probabilities
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1]) # Assume second column is positive class
        else: # Multi-class, need to binarize or choose a class
            # For simplicity, if multi-class, we'll plot ROC for the first class vs. rest
            # A more robust solution would allow selecting a class or plotting all.
            logging.warning("Multi-class ROC curve requested. Plotting for first class vs. rest.")
            # Binarize y_true for the first class
            y_true_bin = (y_true == sorted(np.unique(y_true))[0]).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba[:, 0])

        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=FIG_SIZE)
        sns.set_style("whitegrid")
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('True Positive Rate', fontsize=LABEL_FONT_SIZE)
        plt.title('Receiver Operating Characteristic', fontsize=TITLE_FONT_SIZE, weight='bold')
        plt.legend(loc="lower right", fontsize=LEGEND_FONT_SIZE)
        plt.tight_layout()
        logging.info("ROC curve generated successfully.")
        return plt.gcf(), None
    except Exception as e:
        logging.error(f"Error generating ROC curve: {e}", exc_info=True)
        return None, f"Error generating ROC curve: {e}"

def plot_feature_importance(model, feature_names):
    """Generates a feature importance bar plot for tree-based models.

    Args:
        model: A trained model with a 'feature_importances_' attribute.
        feature_names (list): List of feature names corresponding to the importances.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info("Generating feature importance plot.")
    if not hasattr(model, 'feature_importances_'):
        logging.error("Model does not have feature importances attribute.")
        return None, "Model does not have feature importances."
    
    try:
        importances = model.feature_importances_
        # Sort features by importance in descending order
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=FIG_SIZE)
        sns.set_style("whitegrid")
        
        # Plot top N features for clarity
        num_features_to_plot = min(len(feature_names), 20) # Plot top 20 features or fewer if less available
        
        plt.title("Feature Importances", fontsize=TITLE_FONT_SIZE, weight='bold')
        sns.barplot(x=importances[indices[:num_features_to_plot]], y=[feature_names[i] for i in indices[:num_features_to_plot]], palette="viridis")
        plt.xlabel("Relative Importance", fontsize=LABEL_FONT_SIZE)
        plt.ylabel("Feature Name", fontsize=LABEL_FONT_SIZE)
        plt.tight_layout()
        logging.info("Feature importance plot generated successfully.")
        return plt.gcf(), None
    except Exception as e:
        logging.error(f"Error generating feature importance plot: {e}", exc_info=True)
        return None, f"Error generating feature importance plot: {e}"

def plot_elbow_curve(X, max_k=10):
    """Generates an elbow curve to help determine the optimal number of clusters (k) for KMeans.

    Args:
        X (pd.DataFrame or np.array): The input data for clustering.
        max_k (int, optional): The maximum number of clusters to test. Defaults to 10.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info(f"Generating elbow curve for max_k={max_k}")
    inertias = []
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X) # Ensure X is a DataFrame for .dropna()
    X_cleaned = X.dropna() # Handle NaNs for KMeans
    
    if X_cleaned.empty:
        logging.error("Data is empty after cleaning for Elbow Curve.")
        return None, "Data is empty after cleaning for Elbow Curve."

    # Ensure max_k is not greater than the number of samples
    if max_k > len(X_cleaned):
        logging.warning(f"max_k ({max_k}) is greater than number of samples ({len(X_cleaned)}). Adjusting max_k.")
        max_k = len(X_cleaned)
    
    if max_k < 1:
        return None, "max_k must be at least 1."

    try:
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init to suppress warning
            kmeans.fit(X_cleaned)
            inertias.append(kmeans.inertia_)
        
        plt.figure(figsize=FIG_SIZE)
        sns.set_style("whitegrid")
        plt.plot(range(1, max_k + 1), inertias, marker='o', linestyle='-', color=PRIMARY_COLOR)
        plt.xlabel('Number of clusters (k)', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('Inertia', fontsize=LABEL_FONT_SIZE)
        plt.title('Elbow Method For Optimal k', fontsize=TITLE_FONT_SIZE, weight='bold')
        plt.xticks(np.arange(1, max_k + 1, 1)) # Ensure integer ticks
        plt.tight_layout()
        logging.info("Elbow curve generated successfully.")
        return plt.gcf(), None
    except Exception as e:
        logging.error(f"Error generating elbow curve: {e}", exc_info=True)
        return None, f"Error generating elbow curve: {e}"

def plot_cluster_plot(X, labels, title="Cluster Plot"):
    """Generates a 2D scatter plot of clusters, optionally after dimensionality reduction.

    Args:
        X (pd.DataFrame or np.array): The input data.
        labels (array-like, optional): Cluster labels for coloring points. If None, points are not colored.
        title (str, optional): Title of the plot. Defaults to "Cluster Plot".

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info(f"Generating cluster plot with title: {title}")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Handle NaNs before dimensionality reduction
    X_cleaned = X.dropna()
    if X_cleaned.empty:
        logging.error("Data is empty after cleaning for Cluster Plot.")
        return None, "Data is empty after cleaning for Cluster Plot."

    plot_df = X_cleaned.copy()
    xlabel = 'Feature 1'
    ylabel = 'Feature 2'

    # Reduce dimensions to 2 if data has more than 2 features
    if X_cleaned.shape[1] > 2:
        try:
            logging.info("Applying PCA for dimensionality reduction to 2 components.")
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X_cleaned)
            plot_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])
            xlabel = 'Principal Component 1'
            ylabel = 'Principal Component 2'
        except Exception as e:
            logging.error(f"Could not reduce dimensions for cluster plot using PCA: {e}", exc_info=True)
            return None, f"Could not reduce dimensions for cluster plot: {e}"
    elif X_cleaned.shape[1] == 1:
        logging.error("Data must have at least 2 dimensions for a 2D cluster plot.")
        return None, "Data must have at least 2 dimensions for a 2D cluster plot."

    plt.figure(figsize=FIG_SIZE)
    sns.set_style("whitegrid")
    
    if labels is not None:
        # Align labels with cleaned data if necessary
        if isinstance(labels, pd.Series):
            labels_aligned = labels.loc[X_cleaned.index] if labels.index.equals(X.index) else labels # Simple alignment
        else:
            labels_aligned = labels # Assume already aligned or numpy array
        sns.scatterplot(x=plot_df.iloc[:, 0], y=plot_df.iloc[:, 1], hue=labels_aligned, palette='viridis', s=50, alpha=0.7)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=LEGEND_FONT_SIZE)
    else:
        sns.scatterplot(x=plot_df.iloc[:, 0], y=plot_df.iloc[:, 1], s=50, alpha=0.7, color=PRIMARY_COLOR)
        
    plt.title(title, fontsize=TITLE_FONT_SIZE, weight='bold')
    plt.xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    plt.ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    plt.tight_layout()
    logging.info("Cluster plot generated successfully.")
    return plt.gcf(), None

def plot_dendrogram(X):
    """Generates a dendrogram for hierarchical clustering.

    Args:
        X (pd.DataFrame or np.array): The input data for clustering.

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info("Generating dendrogram.")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X_cleaned = X.dropna() # Handle NaNs
    
    if X_cleaned.empty:
        logging.error("Data is empty after cleaning for Dendrogram.")
        return None, "Data is empty after cleaning for Dendrogram."

    # Limit the number of samples for dendrogram for performance and readability
    if X_cleaned.shape[0] > 1000: 
        logging.warning(f"Dendrogram data size ({X_cleaned.shape[0]}) is large. Sampling 1000 points.")
        X_cleaned = X_cleaned.sample(n=1000, random_state=42)

    try:
        linked = linkage(X_cleaned, 'ward') # Ward method minimizes variance within clusters
        plt.figure(figsize=(12, 8))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, leaf_rotation=90, leaf_font_size=8)
        plt.title('Hierarchical Clustering Dendrogram', fontsize=TITLE_FONT_SIZE, weight='bold')
        plt.xlabel('Sample Index or Cluster Size', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('Distance', fontsize=LABEL_FONT_SIZE)
        plt.tight_layout()
        logging.info("Dendrogram generated successfully.")
        return plt.gcf(), None
    except Exception as e:
        logging.error(f"Error generating dendrogram: {e}", exc_info=True)
        return None, f"Error generating dendrogram: {e}"

def plot_tsne(X, labels=None):
    """Generates a t-SNE plot for dimensionality reduction and visualization of high-dimensional data.

    Args:
        X (pd.DataFrame or np.array): The input high-dimensional data.
        labels (array-like, optional): Labels for coloring points (e.g., cluster assignments).

    Returns:
        tuple: A matplotlib Figure object and an error message (None if successful).
    """
    logging.info("Generating t-SNE plot.")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X_cleaned = X.dropna() # Handle NaNs
    
    if X_cleaned.empty:
        logging.error("Data is empty after cleaning for t-SNE.")
        return None, "Data is empty after cleaning for t-SNE."

    # t-SNE can be computationally expensive on large datasets, consider sampling
    if X_cleaned.shape[0] > 2000:
        logging.warning(f"t-SNE data size ({X_cleaned.shape[0]}) is large. Sampling 2000 points.")
        X_cleaned = X_cleaned.sample(n=2000, random_state=42)
        if labels is not None:
            # Align labels with sampled data
            if isinstance(labels, pd.Series):
                labels = labels.loc[X_cleaned.index]
            else: # If numpy array, convert to series for easy indexing
                labels = pd.Series(labels).loc[X_cleaned.index]

    try:
        # Perplexity should be less than the number of samples
        perplexity_val = min(30, len(X_cleaned) - 1) if len(X_cleaned) > 1 else 1
        if perplexity_val < 1:
            return None, "Not enough samples for t-SNE (need at least 2)."

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
        X_tsne = tsne.fit_transform(X_cleaned)
        
        plt.figure(figsize=FIG_SIZE)
        sns.set_style("whitegrid")
        
        if labels is not None:
            sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='viridis', s=50, alpha=0.7)
            plt.legend(title='Cluster/Label', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=LEGEND_FONT_SIZE)
        else:
            sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], s=50, alpha=0.7, color=PRIMARY_COLOR)
            
        plt.title('t-SNE Plot', fontsize=TITLE_FONT_SIZE, weight='bold')
        plt.xlabel('t-SNE Component 1', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('t-SNE Component 2', fontsize=LABEL_FONT_SIZE)
        plt.tight_layout()
        logging.info("t-SNE plot generated successfully.")
        return plt.gcf(), None
    except Exception as e:
        logging.error(f"Error generating t-SNE plot: {e}", exc_info=True)
        return None, f"Error generating t-SNE plot: {e}"
