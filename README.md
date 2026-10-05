
# Income Classification Using Machine Learning

Predicts whether an individual earns more than $50,000 annually using demographic and socioeconomic attributes (such as age, education level, occupation, marital status, and weekly working hours) from the UCI Census Income (Adult) dataset.

## Key Engineering Highlights

* **Zero Data Leakage Pipeline:** Preprocessing (KNN imputation, oversampling, and scaling) strictly respects train/validation boundaries to ensure uncompromised test integrity.
* **Algorithm Implementation from Scratch:** Implemented Logistic Regression via vectorised Gradient Descent entirely in NumPy without relying on `scikit-learn` modeling wrappers.
* **Ensemble Architecture:** Constructed a weighted probability ensemble combining Decision Trees, KNN, and Custom Logistic Regression to lower false negatives by ~27%.

---

## Results at a Glance

All models were trained on the oversampled training set (49,440 rows) and evaluated on an independent, untouched test set (16,281 rows; 12,435 ≤$50K and 3,846 >$50K).

| Model | Accuracy | Precision (>$50K) | Recall (>$50K) |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | 81.62% | 0.61 | 0.62 |
| **Logistic Regression** (from scratch) | 80.58% | 0.56 | 0.85 |
| **KNN** ($k=5$) | 77.37% | 0.51 | 0.77 |
| **Weighted Ensemble** | **82.21%** | 0.60 | 0.72 |

The ensemble achieved the highest accuracy ($82.21\%$) and optimal precision-recall balance, significantly reducing false negatives compared to standalone models (1,085 vs. 1,480 high earners missed).

---

## Method & Architecture

### 1. Data Cleaning & KNN Imputation
Missing values (represented as `?`) were identified in attributes like `workclass` (5.64%), `occupation` (5.66%), and `native-country` (1.79%).
* Continuous features were min-max scaled and passed to a **KNN Imputer** ($k=5$) before restoring original scale.
* Categorical attributes were imputed using mode replacement.

### 2. Class Imbalance & Data Leakage Prevention
The raw dataset is naturally imbalanced (~76% ≤$50K and ~24% >$50K). 
* To guarantee test set integrity, **data splitting is performed prior to oversampling**. 
* **Random Oversampling** is applied exclusively to the training split, bringing class counts to balance without duplicating validation or test samples.

### 3. Encoding & Feature Scaling
* **Continuous features:** Standardized using z-score scaling: $(x - \mu) / \sigma$.
* **Categorical features:** One-Hot Encoded (`handle_unknown='ignore'`).
* Scalers and transformers are fit strictly on the training partition and applied downstream to test data to prevent data snooping.

### 4. Model Architectures
* **Decision Tree:** Baseline non-linear classifier (`scikit-learn`).
* **Custom Logistic Regression:** Custom NumPy implementation utilizing the Sigmoid function and vectorised Gradient Descent (learning rate $\alpha=0.05$, $5,000$ iterations).
* **K-Nearest Neighbors:** Non-parametric classifier ($k=5$, Euclidean distance metric).
* **Weighted Ensemble:** Probability-weighted average using weights derived from relative model accuracies ($w_{DT}=0.32$, $w_{LR}=0.33$, $w_{KNN}=0.35$).

---

## Repository File Structure

| File | Description |
| :--- | :--- |
| `income_classification.ipynb` | Interactive Jupyter Notebook containing full EDA, pipeline execution, and model performance visualizations. |
| `income_classification.py` | Modular Python script executing the full data pipeline and model evaluators. |
| `census-income.data.csv` | Raw training dataset. |
| `census-income.test.csv` | Raw test dataset. |
| `requirements.txt` | Python dependencies list. |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run via script
python income_classification.py

# Or run interactively in Jupyter
jupyter notebook income_classification.ipynb
