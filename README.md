# Academic Success Predictor

This project predicts students' **academic outcomes** (Dropout, Enrolled, Graduate) using university demographic and performance data. It supports **two model alternatives** — XGBoost and CatBoost — and includes a clean visualization/reporting pipeline.

---

## Problem Statement

Given labeled student records, the task is to build a multi-class classification model that predicts whether a student will:

- Graduate  
- Remain Enrolled  
- Drop Out  

This is framed as a **supervised multi-class classification** problem.

---

## Project Structure

```
.
├── data/                      # Input CSV files and output submission
├── model/                    # Saved model files
├── outputs/                  # Visualizations (per model)
├── src/
│   ├── explore_data.py       # EDA, correlation, feature prep
│   ├── main.py               # Entry point with model toggle and full pipeline
│   ├── models/
│   │   ├── xgb_model.py      # XGBoost training & submission logic
│   │   └── catboost_model.py # CatBoost training & submission logic
│   ├── visualization.py      # All plots with model name tagging
├── README.md
```

---

## Environment

- **Python**: 3.11
- **IDE**: PyCharm (Windows)
- **Main Libraries**:  
  `xgboost`, `catboost`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`

---

## Features & Highlights

- **Supports both XGBoost & CatBoost models**
- Handles **all features as categorical**
- Rich set of visualizations: target distribution, count plots, feature importance, correlation matrix, confusion matrix
- Cross-validation (StratifiedKFold)
- Organized outputs per model
- Final report (TXT) with metrics, data overview, and model config
- No need for feature scaling or imputation logic for modeling — handled by CatBoost/XGBoost

---

## Model Details

### XGBoost (XGBClassifier)
- Encoding: LabelEncoded categories
- CV Metrics: Accuracy, F1-Macro
- Output: `model/xgb_model.joblib`, `data/sample_submission_xgb.csv`

### CatBoost (CatBoostClassifier)
- Encoding: Raw string-based categorical features
- Built-in order-aware target encoding
- Output: `model/catboost_model.cbm`, `data/sample_submission_catboost.csv`

---

## Example Outputs

Each model generates:
- Count plots of categorical variables
- Feature importance charts
- Confusion matrix
- Saved under `outputs/{model_name}/`

---

## How to Run

### Run Full Pipeline

```bash
cd src
python main.py
```

Toggle model in `main.py`:
```python
MODEL_CHOICE = "xgb"  # or "catboost"
```

---

## Submission File

The output file follows this format:
```csv
id,Target
1234,Graduate
1235,Dropout
...
```

Saved as:
- `data/sample_submission_xgb.csv`
- or `data/sample_submission_catboost.csv`

---

## Final Report

Each run ends with a `final_report.txt` under the appropriate `outputs/{model_name}/` folder, including:

- Model used
- Train/test shape
- Target distribution
- Classification report
- Execution time

---

## To Do (Ideas)

- [ ] Add SHAP explanation for interpretability
- [ ] Add hyperparameter tuning script
- [ ] Package as a CLI or streamlit app

---

