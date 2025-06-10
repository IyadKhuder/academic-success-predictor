# Academic Success Prediction (Multi-Class Classification)

This project predicts the academic outcome of university students: whether they **graduate**, **drop out**, or **remain enrolled**. It uses demographic and academic performance data and applies an **XGBoost classifier**.

## Project Structure
- `explore_data.py`: Main script that performs EDA, feature engineering, model training, and evaluation.
- `data/`: Contains input files and generated submission.
- `requirements.txt`: Dependencies used.
- `README.md`: Project overview.

## Model
- Type: Multi-class classification
- Algorithm: XGBoost (with cross-validation)
- Metrics: Accuracy, F1-macro, Confusion Matrix, Feature Importance

## ⚙Environment
- Python 3.11
- IDE: PyCharm (Windows)
- Libraries: `xgboost`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`

## Run
```bash
python explore_data.py
