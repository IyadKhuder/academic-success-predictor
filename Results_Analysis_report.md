# Model Comparison Analysis: XGBoost vs CatBoost

## Overview

This document summarizes the performance, execution time, and behavioral analysis of two models — **CatBoost** and **XGBoost** — trained on the academic success classification task.

---

## Model Comparison: XGBoost vs CatBoost

| Metric            | CatBoost | XGBoost | Better     |
|-------------------|----------|---------|------------|
| Accuracy          | 0.84     | 0.88    |  XGBoost |
| Macro F1-score    | 0.80     | 0.85    |  XGBoost |
| Weighted F1-score | 0.83     | 0.88    |  XGBoost |

---

## Class-wise Breakdown

### 1. **Dropout**
- **CatBoost**:
  - Precision: 0.91
  - Recall: 0.83
  - F1-score: 0.87
- **XGBoost**:
  - Precision: 0.94
  - Recall: 0.88
  - F1-score: 0.90 

### 2. **Enrolled**
- **CatBoost**:
  - Precision: 0.66
  - Recall: 0.63
  - F1-score: 0.64
- **XGBoost**:
  - Precision: 0.76
  - Recall: 0.71
  - F1-score: 0.73 

### 3. **Graduate**
- **CatBoost**:
  - Precision: 0.86
  - Recall: 0.93
  - F1-score: 0.89
- **XGBoost**:
  - Precision: 0.88
  - Recall: 0.95
  - F1-score: 0.91 

---

## Execution Time

| Model     | Total Time (s) |
|-----------|----------------|
| CatBoost  | 1237.19 s      |
| XGBoost   | 18.51 s       |

XGBoost was **~66× faster** in execution time.

---

## Why Did XGBoost Outperform CatBoost?

It was initially expected that **CatBoost** would perform better due to the nature of the features:
- Many features are **categorical in essence**, though they appear as numeric types (`float64`, `int64`).
- Examples include **course codes**, **parental education levels**, etc., which are **non-ordinal categorical values**.

However, XGBoost still outperformed, and the reasons likely include:

1. **Effective Greedy Splitting**: XGBoost’s splitting strategy may still capture useful signal from numerically-encoded categories.
2. **Noise Tolerance**: It handled non-semantic numeric values robustly, especially given the dataset size.
3. **Optimization**: Its highly optimized computation (multi-threading, efficient trees) leads to much faster execution.

---

## Final Conclusion

Despite the theoretical advantage of CatBoost for categorical features:
- **XGBoost was both faster and more accurate**.
- It achieved better **F1-scores** across all classes, including the minority class “Enrolled”.
- The overall accuracy and training efficiency make XGBoost a better choice for this project at its current stage.

Iyad Khuder
---
