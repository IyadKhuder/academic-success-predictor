import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import plot_importance
import os

def save_fig(name, folder="outputs", model_name=None):
    if model_name:
        folder = os.path.join(folder, model_name.lower())
    os.makedirs(folder, exist_ok=True)

    final_name = f"{model_name}_{name}" if model_name else name
    plt.tight_layout()
    path = os.path.join(folder, f"{final_name}.png")
    plt.savefig(path)
    plt.close()
    print(f"Figure saved: {path}")


def plot_target_distribution(df, target_column="Target", model_name=None):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=target_column, order=df[target_column].value_counts().index)
    plt.title(f"Target Distribution ({model_name})" if model_name else "Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Count")
    save_fig("target_distribution", model_name)


def plot_top_variance_histograms(df, numerical_features, top_n=6, target_column=None, model_name=None):
    variances = df[numerical_features].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()

    for feature in top_features:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=feature, order=df[feature].value_counts().index)
        plt.title(f"Distribution of {feature} ({model_name})" if model_name else f"Distribution of {feature}")
        plt.xticks(rotation=45)
        save_fig(f"categorical_distribution_{feature}", model_name)


def plot_correlation_heatmap(df, numerical_features, model_name=None):
    plt.figure(figsize=(12, 10))
    corr_matrix = df[numerical_features].corr()
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
    plt.title(f"Correlation Matrix (Numerical Features) ({model_name})" if model_name else "Correlation Matrix (Numerical Features)")
    save_fig("correlation_heatmap", model_name)


def plot_feature_boxplots(df, features, target_column="Target", model_name=None):
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=target_column, y=feature)
        plt.title(f"{feature} Distribution by {target_column} ({model_name})" if model_name else f"{feature} Distribution by {target_column}")
        save_fig(f"boxplot_{feature}", model_name)


def plot_confusion_matrix(y_true, y_pred, labels, model_name=None):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", values_format="d"
    )
    plt.title(f"Confusion Matrix ({model_name})" if model_name else "Confusion Matrix")
    save_fig("confusion_matrix", model_name)


def plot_xgb_feature_importance(model, max_num_features=15, model_name=None):
    plt.figure(figsize=(10, 8))
    plot_importance(model, max_num_features=max_num_features, importance_type="gain", show_values=False)
    plt.title(f"Top Feature Importances (XGBoost) ({model_name})" if model_name else "Top Feature Importances (XGBoost)")
    save_fig("feature_importance", model_name)


def plot_catboost_feature_importance(model, feature_names, max_num_features=15, model_name=None):
    importances = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(max_num_features)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x="Importance", y="Feature")
    plt.title(f"Top Feature Importances (CatBoost) ({model_name})" if model_name else "Top Feature Importances (CatBoost)")
    save_fig("feature_importance", model_name)
