import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold

def prepare_data():
    # ============================
    # Step 1: Data Loading & Inspection
    # ============================

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\n--- Train Info ---")
    print(train_df.info())

    print("\n--- Test Info ---")
    print(test_df.info())

    print("\n--- Missing Values (Train) ---")
    print(train_df.isnull().sum())

    print("\n--- Missing Values (Test) ---")
    print(test_df.isnull().sum())

    print("\n--- Unique Values in Train Columns ---")
    print(train_df.nunique())

    print("\n--- Data Types (Train) ---")
    print(train_df.dtypes)

    print("\n--- Target Distribution ---")
    print(train_df["Target"].value_counts(normalize=True))

    # ============================
    # Step 2: Feature Classification
    # ============================

    numerical_features = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_features = [col for col in numerical_features if col != "id"]

    categorical_features = train_df.select_dtypes(include=["object", "category"]).columns.tolist()

    print("\n--- Numerical Features (excluding 'id') ---")
    print(numerical_features)

    print("\n--- Categorical Features ---")
    print(categorical_features)

    # ============================
    # Step 3: Handle Missing Values
    # ============================

    def fill_missing(df):
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["float64", "int64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df

    train_df = fill_missing(train_df)
    test_df = fill_missing(test_df)

    # ============================
    # Step 4: Correlation Check
    # ============================

    corr_matrix = train_df[numerical_features].corr()
    upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
    high_corr_pairs = [(col, row, upper_tri.loc[row, col])
                       for col in upper_tri.columns
                       for row in upper_tri.index
                       if upper_tri.loc[row, col] > 0.85]

    print("\n--- Highly Correlated Feature Pairs (r > 0.85) ---")
    if high_corr_pairs:
        for f1, f2, corr in high_corr_pairs:
            print(f"{f1} <--> {f2} | Correlation: {corr:.2f}")
    else:
        print("No feature pairs exceed correlation threshold.")

    # ============================
    # Step 5: Feature Engineering
    # ============================

    label_encoder = LabelEncoder()
    train_df["Target_encoded"] = label_encoder.fit_transform(train_df["Target"])

    print("\nTarget class mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    selector = VarianceThreshold(threshold=0.0)
    selector.fit(train_df[numerical_features])
    low_variance_features = [col for col, keep in zip(numerical_features, selector.get_support()) if not keep]

    if low_variance_features:
        print("Low-variance features to consider dropping:", low_variance_features)
    else:
        print("No constant features found.")

    model_features = [col for col in numerical_features if col not in low_variance_features]

    X_train = train_df[model_features].copy()
    y_train = train_df["Target_encoded"]
    X_test = test_df[model_features].copy()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    return train_df, test_df, X_train, y_train, X_test, numerical_features
