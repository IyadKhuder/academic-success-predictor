import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

def train_model():
    # ============================
    # Step 1: Load Processed Data
    # ============================
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    model_features = [col for col in train_df.columns
                      if col not in ["id", "Target"] and train_df[col].dtype in ["float64", "int64"]]

    X_train = train_df[model_features].copy()
    X_test = test_df[model_features].copy()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["Target"])

    # ============================
    # Step 2: Define & Train Model
    # ============================
    xgb_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Accuracy CV
    cv_accuracy = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")

    # F1 Macro CV
    cv_f1 = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"CV Macro F1-score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Final fit
    xgb_model.fit(X_train, y_train)

    # ============================
    # Step 3: Evaluation & Reporting
    # ============================
    train_preds = xgb_model.predict(X_train)
    print("\nClassification Report on Full Training Set:")
    print(classification_report(y_train, train_preds, target_names=label_encoder.classes_))

    # ============================
    # Step 4: Test Prediction & Submission
    # ============================
    test_preds = xgb_model.predict(X_test)
    test_labels = label_encoder.inverse_transform(test_preds)

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "Target": test_labels
    })

    submission_path = "data/sample_submission_generated.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSample submission saved to '{submission_path}'")

    # ============================
    # Step 5: Save Model
    # ============================
    os.makedirs("model", exist_ok=True)
    joblib.dump(xgb_model, "model/xgb_model.joblib")

    return xgb_model, y_train, label_encoder
