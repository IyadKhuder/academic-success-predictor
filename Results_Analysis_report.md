# Model Training Summary (XGBoost)

## Data
- Train shape: (76520, 38)
- Test shape: (19130, 37)

## Features
- Numerical features: auto-detected from train set (excluding 'id', 'Target')
- Low-variance features: dropped if variance == 0

## Target Distribution
The distribution of target classes in the training set was analyzed visually and statistically.

## Cross-Validation Results
- Accuracy (mean ± std): Approx. reported in logs using StratifiedKFold (5 splits)
- F1-macro (mean ± std): As above

## Final Training Evaluation
```
Classification Report on Full Training Set:
               precision    recall  f1-score   support

     Dropout       ...        ...      ...       ...
     Enrolled      ...        ...      ...       ...
     Graduate      ...        ...      ...       ...

    accuracy                           ...      ...
   macro avg       ...        ...      ...
weighted avg       ...        ...      ...
```

## Model Artifacts
- Trained model saved to: `model/xgb_model.joblib`
- Submission file generated at: `data/sample_submission_generated.csv`
- Visual outputs saved in: `outputs/` folder:
  - Target distribution
  - Histograms of top variance features
  - Correlation heatmap
  - Boxplots by target
  - Confusion matrix
  - Feature importance (gain-based)
