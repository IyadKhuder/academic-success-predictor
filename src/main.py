import os

from explore_data import prepare_data
import time

from visualization import (
    plot_target_distribution,
    plot_top_variance_histograms,
    plot_correlation_heatmap,
    plot_feature_boxplots,
    plot_confusion_matrix,
    plot_xgb_feature_importance,
    plot_catboost_feature_importance,
)
import warnings
warnings.filterwarnings("ignore")  # optional, to suppress verbose output

# Toggle model here: "xgb" or "catboost"
MODEL_CHOICE = "catboost"  # or "xgb"
model_name = MODEL_CHOICE.upper()

if MODEL_CHOICE == "xgb":
    from models.xgb_model import train_model
elif MODEL_CHOICE == "catboost":
    from models.catboost_model import train_model
else:
    raise ValueError("Unsupported model choice. Use 'xgb' or 'catboost'.")

if __name__ == "__main__":
    start_time = time.time()
    train_df, test_df, X_train, y_train, X_test, numerical_features = prepare_data()

    if MODEL_CHOICE == "catboost":
        X_train = X_train.astype(str)

    model, y_encoded, label_encoder = train_model()

    # === Visualizations ===
    plot_target_distribution(train_df, target_column="Target", model_name=model_name)
    plot_top_variance_histograms(train_df, numerical_features, target_column="Target", model_name=model_name)
    plot_correlation_heatmap(train_df, numerical_features, model_name=model_name)
    plot_feature_boxplots(train_df, numerical_features[:3], target_column="Target", model_name=model_name)

    y_pred_train = model.predict(X_train)
    y_pred_train = y_pred_train.flatten() if hasattr(y_pred_train, "flatten") else y_pred_train

    plot_confusion_matrix(y_encoded, y_pred_train, label_encoder.classes_, model_name=model_name)

    if MODEL_CHOICE == "xgb":
        plot_xgb_feature_importance(model, model_name=model_name)
    elif MODEL_CHOICE == "catboost":
        plot_catboost_feature_importance(model, X_train.columns.tolist(), model_name=model_name)

    # === Final Summary Report ===
    print("\n" + "=" * 40)
    print("FINAL MODEL REPORT")
    print("=" * 40)
    print(f"Model used: {model_name}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Number of features used: {X_train.shape[1]}")
    print("\nTarget distribution (train):")
    print(train_df["Target"].value_counts(normalize=True).round(3))
    print("\nClassification Report (Train Set):")
    from sklearn.metrics import classification_report
    print(classification_report(y_encoded, y_pred_train, target_names=label_encoder.classes_))
    print("=" * 40)
    duration = time.time() - start_time
    report_lines = [
        "=" * 40,
        "FINAL MODEL REPORT",
        "=" * 40,
        f"Model used: {model_name}",
        f"Train shape: {train_df.shape}",
        f"Test shape: {test_df.shape}",
        f"Number of features used: {X_train.shape[1]}",
        "\nTarget distribution (train):",
        train_df["Target"].value_counts(normalize=True).round(3).to_string(),
        "\nClassification Report (Train Set):",
        classification_report(y_encoded, y_pred_train, target_names=label_encoder.classes_),
        "=" * 40,

        f"\nTotal execution time: {duration:.2f} seconds"
    ]

    report_path = f"outputs/{model_name.lower()}/final_report.txt"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"📄 Report saved to: {report_path}")


print(f"\nTotal execution time: {duration:.2f} seconds")
