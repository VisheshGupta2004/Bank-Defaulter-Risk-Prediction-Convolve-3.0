# Convolve 3.0 — Credit Default Risk Prediction

Competition project by **Team Neural Thinkers** for Convolve 3.0. The project predicts the probability of credit default from account, transaction, and credit bureau features, then converts the probability into a behavior score.

## Project overview

The development data described in the report contains 96,806 rows and 1,216 columns. The validation data contains 41,792 rows and 1,215 columns. The target is `bad_flag`; `account_number` is an identifier and is excluded from model features.

The notebook explores missingness and feature cardinality, drops columns with more than 30% missing values and constant columns, then imputes remaining missing values with the mode for selected categorical features and the median otherwise. It compares LightGBM, XGBoost, CatBoost, and a five-model LightGBM ensemble. The notebook reports ROC AUC, PR AUC, classification metrics, and a confusion matrix. Validation probabilities are transformed as:

```text
behavior_score = (1 - predicted_probability_of_default) * 100
```

A higher score therefore indicates lower predicted default risk.

## Repository contents

- `Convolve_3.ipynb` — exploratory analysis, preprocessing, model training, validation scoring, and behavior score calculation.
- `Convolve_3_report.pdf` — team report describing the problem, data, methods, and results.
- `final_validation_predictions.csv` — local competition output; excluded from Git because it contains account identifiers and row-level predictions.

## Run the notebook

1. Obtain the competition development and validation data from the competition source. They are not included in this repository.
2. Install the Python packages listed in `requirements.txt`.
3. Update the two CSV paths near the start of the notebook to point to your local copies:
   - `Dev_data_to_be_shared.csv`
   - `validation_data_to_be_shared.csv`
4. Run the notebook from top to bottom.

The original notebook uses Google Colab paths (`/content/...`) and includes a `pip install catboost` cell; update the paths for your environment. The notebook currently evaluates models on a stratified holdout split and uses a fixed 0.5 classification threshold for reported class metrics.

## Notes

- Competition data and generated account-level prediction files are intentionally excluded from version control. Do not commit private or competition-restricted data.
- The validation CSV in the working folder is ignored by Git; it remains available locally.
- The notebook's final validation probabilities are produced by the five-model LightGBM ensemble. The earlier single-model prediction cell is an intermediate result.
- The report and notebook are the project's source artifacts; reproduce results with the original competition data and the notebook's specified preprocessing.
