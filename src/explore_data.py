import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import plot_importance

# ============================
# Step 1: Data Loading
# ============================

# Paths to your CSV files (adjust paths as needed)
train_path = "data/train.csv"
test_path = "data/test.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# =======================================
# Step 2: Data Exploration & Profiling
# =======================================

# Inspect shapes
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Show structure summary
print("\n--- Train Info ---")
print(train_df.info())
print("\n--- Test Info ---")
print(test_df.info())

# Missing values count
print("\n--- Missing Values (Train) ---")
print(train_df.isnull().sum())

print("\n--- Missing Values (Test) ---")
print(test_df.isnull().sum())

# Unique values per column (Train)
print("\n--- Unique Values in Train Columns ---")
print(train_df.nunique())

# Data types (Train)
print("\n--- Data Types (Train) ---")
print(train_df.dtypes)

# Class balance in Target column
print("\n--- Target Distribution ---")
print(train_df["Target"].value_counts(normalize=True))

# ============================
# Step 3: Feature Analysis
# ============================

# Identify numerical and categorical features (exclude 'id')
numerical_features = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_features = [col for col in numerical_features if col != "id"]

categorical_features = train_df.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n--- Numerical Features (excluding 'id') ---")
print(numerical_features)

print("\n--- Categorical Features ---")
print(categorical_features)

print("\n--- First 5 Rows of Training Data ---")
print(train_df.head())

# ============================
# Step 4: Data Cleaning
# ============================

# Handle missing values in train_df
print("\n--- Cleaning: Handling Missing Values in Train ---")
missing_train = train_df.isnull().sum()
missing_train = missing_train[missing_train > 0]
print(missing_train)

for col in missing_train.index:
    missing_count = train_df[col].isnull().sum()
    if train_df[col].dtype in ["float64", "int64"]:
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        print(f"[Train] Replaced {missing_count} missing values in '{col}' with median: {median_val}")
    else:
        mode_val = train_df[col].mode()[0]
        train_df[col].fillna(mode_val, inplace=True)
        print(f"[Train] Replaced {missing_count} missing values in '{col}' with mode: {mode_val}")
if missing_train.empty:
    print("[Train] The training data is clean; no replacements needed.")

# Handle missing values in test_df
print("\n--- Cleaning: Handling Missing Values in Test ---")
missing_test = test_df.isnull().sum()
missing_test = missing_test[missing_test > 0]
print(missing_test)

for col in missing_test.index:
    missing_count = test_df[col].isnull().sum()
    if test_df[col].dtype in ["float64", "int64"]:
        median_val = test_df[col].median()
        test_df[col].fillna(median_val, inplace=True)
        print(f"[Test] Replaced {missing_count} missing values in '{col}' with median: {median_val}")
    else:
        mode_val = test_df[col].mode()[0]
        test_df[col].fillna(mode_val, inplace=True)
        print(f"[Test] Replaced {missing_count} missing values in '{col}' with mode: {mode_val}")
if missing_test.empty:
    print("[Test] The test data is clean; no replacements needed.")

# ============================
# Step 5: Exploratory Data Analysis (EDA)
# ============================

print("\n--- Step 5: Exploratory Data Analysis (EDA) ---")


# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x="Target", order=train_df["Target"].value_counts().index)
plt.title("Target Distribution")
plt.xlabel("Target Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2. Histograms for top 6 numerical features (by variance)
num_variances = train_df[numerical_features].var().sort_values(ascending=False)
top_numerical = num_variances.head(6).index.tolist()

train_df[top_numerical].hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle("Top 6 High-Variance Numerical Features", fontsize=14)
plt.tight_layout()
plt.show()

# 3. Correlation matrix (numerical)
plt.figure(figsize=(12, 10))
corr_matrix = train_df[numerical_features].corr()
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Matrix (Numerical Features)")
plt.tight_layout()
plt.show()


# 4. Boxplots of top 3 features vs. Target
for feature in top_numerical[:3]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_df, x="Target", y=feature)
    plt.title(f"{feature} Distribution by Target")
    plt.tight_layout()
    plt.show()



# 5. Correlation filter (just for review)
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
# Step 6: Feature Engineering
# ============================

print("\n--- Step 6: Feature Engineering ---")

# 1. Encode target variable
label_encoder = LabelEncoder()
train_df["Target_encoded"] = label_encoder.fit_transform(train_df["Target"])
print("Target class mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# 2. Low-variance filter
selector = VarianceThreshold(threshold=0.0)
selector.fit(train_df[numerical_features])
low_variance_features = [col for col, keep in zip(numerical_features, selector.get_support()) if not keep]

if low_variance_features:
    print("Low-variance features to consider dropping:", low_variance_features)
else:
    print("No constant features found.")

# 3. Define feature set
model_features = [col for col in numerical_features if col not in low_variance_features]

# 4. Define training and test sets for modeling
X_train = train_df[model_features].copy()
y_train = train_df["Target_encoded"]
X_test = test_df[model_features].copy()

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ============================
# Step 7: Model Training (XGBoost)
# ============================

print("\n--- Step 7: Model Training with XGBoost ---")

# 1. Define model
xgb_model = XGBClassifier(
    objective="multi:softprob",  # Predict probabilities for multi-class
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# 2. Evaluate with Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # We're using 5-fold stratified cross-validation for balanced evaluation

# Accuracy
cv_accuracy = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="accuracy")
print(f"CV Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")

# Macro F1-score
cv_f1 = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring="f1_macro") #f1_macro is particularly good for imbalanced class distributions.
print(f"CV Macro F1-score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# 3. Fit final model
xgb_model.fit(X_train, y_train)

# 4. Predict on train set for sanity check
train_preds = xgb_model.predict(X_train)
print("\nClassification Report on Full Training Set:")
print(classification_report(y_train, train_preds, target_names=label_encoder.classes_))

# 5. Predict on test set
test_preds = xgb_model.predict(X_test)
test_preds_labels = label_encoder.inverse_transform(test_preds)

# 6. Prepare submission DataFrame
submission_df = pd.DataFrame({
    "id": test_df["id"],
    "Target": test_preds_labels
})

# Save submission to CSV
submission_df.to_csv("data/sample_submission_generated.csv", index=False)
print("\nSample submission saved to 'data/sample_submission_generated.csv'")


# ============================
# Step 8: Final Evaluation & Confusion Matrix
# ============================

print("\n--- Step 8: Final Evaluation ---")

# Confusion matrix
cm = confusion_matrix(y_train, train_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

print("Confusion Matrix (on full training set):")
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Training Set")
plt.tight_layout()
plt.show()


# ============================
# Step 9: Model Interpretation (Feature Importance)
# ============================

print("\n--- Step 9: Feature Importance ---")

# 1. Plot feature importance
plt.figure(figsize=(10, 8))
plot_importance(xgb_model, max_num_features=15, importance_type="gain", show_values=False)
plt.title("Top 15 Feature Importances (by Gain)")
plt.tight_layout()
plt.show()

# 2. Print top 10 feature importances as DataFrame
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Features by Model Importance:")
print(importance_df.head(10).to_string(index=False))
