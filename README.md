# Earthquake Damage Classification

Small project for predicting earthquake damage grades of buildings.

This is based on the DrivenData training competition:

https://www.drivendata.org/competitions/57/nepal-earthquake/

| Model                                  | Score    |
| -------------------------------------- | -------- |
| Competition top leaderboard score      | `0.7558` |
| Best score achieved in this repository | `0.727`  |


The repository includes:

- Feature engineering in Python
- Random Forest baseline notebook
- XGBoost training and tuning notebooks

## Project Structure

- `feature_engineering.py`
- `02-feature_eng.ipynb`
- `03-rand_forest_classifier.ipynb`
- `04-xgb_classifier.ipynb`
- `07-simple_xgb-tuning.ipynb`
- `requirements.txt`

## Data Requirements

Put competition training files in `data/`:

- `data/train_values.csv`
- `data/train_labels.csv`

The script merges them on `building_id`.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Jupyter:

```bash
jupyter notebook
```

Main libraries: pandas, scikit-learn, xgboost, optuna, hyperopt, mlflow, jupyter.
