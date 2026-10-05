# Income Classification Using Machine Learning

Predicts whether an individual earns more than $50,000 per year from demographic and socioeconomic attributes in the UCI Census Income (Adult) dataset (~32,000 training records).

This was a group project for CISC 5790 Data Mining at Fordham University.

## Contents

| File | Description |
|---|---|
| `income_classification.ipynb` | Implementation notebook: preprocessing, models, evaluation, plots |
| `income_classification.py` | Same pipeline as a standalone script (plots are commented out) |
| `census-income.data.csv` | Training data (UCI Adult / Census Income) |
| `census-income.test.csv` | Test data |
| `CISC_Data_Mining_Final_Report.pdf` | Full group report: methodology, results, discussion |
| `requirements.txt` | Python dependencies |

## Pipeline

1. **Missing values:** `?` entries imputed with KNN (k=5) for numeric columns; mode imputation for categorical columns
2. **Class imbalance:** random oversampling of the training set (~76% ≤50K / ~24% >50K originally)
3. **Encoding/scaling:** z-score scaling for continuous features, one-hot encoding for categorical features
4. **Models:**
   - Decision Tree
   - Logistic Regression (implemented from scratch with gradient descent)
   - K-Nearest Neighbors (k=5, Euclidean)
   - Weighted ensemble (weights 0.32 / 0.33 / 0.35, 0.5 threshold)

## Results (external test set, 16,281 records)

| Model | Accuracy |
|---|---|
| Decision Tree | 81.62% |
| Logistic Regression | 80.58% |
| KNN (k=5) | 77.37% |
| Weighted Ensemble | 82.21% |

The test set is not oversampled, so precision on the >50K class is lower than recall for some models. See the report for discussion.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook income_classification.ipynb
# or
python income_classification.py
```

Both expect `census-income.data.csv` and `census-income.test.csv` in the same folder. The data comes from the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult). Running the pipeline also writes `final-train.csv` (about 25 MB), which is not included in the repo.

## Contributors

Ivan Brajkovic, Jose Deleon, Lily Bryan, Paige Mitchell.
The report is a collaborative effort. The notebook contains the preprocessing and model implementation code.
