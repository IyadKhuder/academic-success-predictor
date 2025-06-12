import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
import joblib

def train_model():
    # ============================
    # Step 1: Load Data
    # ============================
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Categorical features (all except ID and Target)
    model_features = [col for col in train_df.columns if col not in ["id", "Target"]]
    cat_features = model_features.copy()

    X_train = train_df[model_features].astype(str).copy()
    X_test = test_df[model_features].astype(str).copy()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["Target"])

    # ============================
    # Step 2: Define & Train Model
    # ============================
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        cat_features=cat_features,
        verbose=False,
        random_seed=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        cat_model, X_train, y_train, cv=cv, scoring="f1_macro"
    )
    print(f"CV Macro F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit
    cat_model.fit(X_train, y_train)

    # ============================
    # Step 3: Evaluation
    # ============================
    train_preds = cat_model.predict(X_train)
    print("\nClassification Report on Full Training Set:")
    print(classification_report(y_train, train_preds, target_names=label_encoder.classes_))

    # ============================
    # Step 4: Generate Submission
    # ============================
    test_preds = cat_model.predict(X_test)
    test_labels = label_encoder.inverse_transform(test_preds.flatten())

    submission_df = pd.DataFrame({
        "id": test_df["id"],
        "Target": test_labels
    })

    submission_path = "data/sample_submission_catboost.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSample submission saved to '{submission_path}'")

    # ============================
    # Step 5: Save Model
    # ============================
    os.makedirs("model", exist_ok=True)
    cat_model.save_model("model/catboost_model.cbm")

    return cat_model, y_train, label_encoder
