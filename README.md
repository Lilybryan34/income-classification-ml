# Income Classification Using Machine Learning

Predicts whether a person earns more than $50,000 a year from demographic and work attributes (age, education, occupation, marital status, hours per week, and so on) using the UCI Census Income (Adult) dataset.


## Results at a glance

All models were trained on the oversampled training set (49,440 rows) and evaluated on the separate, untouched test file (16,281 rows; 12,435 ≤50K and 3,846 >50K).

| Model | Accuracy | Precision (>50K) | Recall (>50K) |
|---|---|---|---|
| Decision Tree | 81.62% | 0.61 | 0.62 |
| Logistic Regression (from scratch) | 80.58% | 0.56 | 0.85 |
| KNN (k=5) | 77.37% | 0.51 | 0.77 |
| Weighted ensemble | **82.21%** | 0.60 | 0.72 |

The ensemble had the best accuracy and the best balance between precision and recall. It also made fewer false negatives than the decision tree alone (1,085 vs. 1,480 high earners missed, about 27% fewer).

## Files

| File | Description |
|---|---|
| `income_classification.ipynb` | Notebook with the full pipeline, models, and plots |
| `income_classification.py` | Same pipeline as a standalone script (plots are commented out) |
| `census-income.data.csv` | Training data |
| `census-income.test.csv` | Test data |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt
jupyter notebook income_classification.ipynb
# or
python income_classification.py
```

Keep the two CSV files in the same folder as the code. Running the pipeline also writes `final-train.csv` (about 25 MB), which is not included in the repo. The KNN imputation step takes about a minute.

## Method

### 1. Missing values
Missing entries are marked `?` in the raw data. In the training file, `workclass` is missing 5.64% of values, `occupation` 5.66%, and `native-country` 1.79% (4,262 missing values in total; the test file has 2,203). These rates are low enough that the columns are kept rather than dropped.

- Numeric columns are imputed with KNN (k=5). The features are one-hot encoded and min-max scaled first so distances are meaningful, then the numeric values are scaled back.
- Categorical columns are filled with the most frequent value (the mode).

### 2. Class imbalance
The training data is about 76% ≤50K (24,720 rows) and 24% >50K (7,841 rows). Random oversampling duplicates minority-class rows until both classes have 24,720. Undersampling would have cut each class to 7,841 and thrown away a lot of data. Oversampling is applied to the training data only. The test set keeps its real distribution.

### 3. Encoding and scaling
- Continuous features: z-score scaling, (x − mean) / std.
- Categorical features: one-hot encoding.
- The scaler and encoder are fit on the training data only and then applied to the test data.

### 4. Models
- **Decision Tree** (scikit-learn, default settings): a baseline that can capture non-linear relationships.
- **Logistic Regression**: written from scratch with NumPy. It uses a sigmoid function and gradient descent (learning rate 0.05, 5,000 iterations) with a 0.5 decision threshold.
- **KNN**: scikit-learn, k=5, Euclidean distance. k=5 was chosen from the accuracy-vs-k plot as a point past the very low values of k, where the model is most likely to overfit. KNN is a "lazy" learner, so `fit` only stores the training data.
- **Weighted ensemble**: averages the predicted probabilities of the three models with weights 0.32 (Decision Tree), 0.33 (Logistic Regression), and 0.35 (KNN), then applies a 0.5 threshold. The weights come from each model's accuracy divided by the sum of the three accuracies, so they are close to equal.

## Detailed results

Rows are actual class, columns are predicted class, in the order ≤50K, >50K.

**Decision Tree**: accuracy 81.62%
```
[[10922  1513]
 [ 1480  2366]]
```
Precision/recall: ≤50K 0.88 / 0.88, >50K 0.61 / 0.62.

**Logistic Regression**: accuracy 80.58%
```
[[9860  2575]
 [ 587  3259]]
```
Precision 0.56, recall 0.85, F1 0.67 for >50K. It catches most high earners but also labels many ≤50K people as >50K.

**KNN (k=5)**: accuracy 77.37%
```
[[9628  2807]
 [ 879  2967]]
```
Precision/recall: ≤50K 0.92 / 0.77, >50K 0.51 / 0.77. The 879 are high earners predicted as ≤50K (false negatives). The 2,807 are ≤50K people predicted as >50K (false positives), which is why precision on >50K is low.

**Weighted ensemble**: accuracy 82.21%
```
[[10623  1812]
 [ 1085  2761]]
```
Precision/recall: ≤50K 0.91 / 0.85, >50K 0.60 / 0.72.

## Takeaways

- Every model is much better at predicting ≤50K than >50K. The >50K class is smaller, so precision on it is low for all models (0.51 to 0.61).
- Logistic Regression and KNN find more high earners (recall 0.85 and 0.77) but at the cost of many false positives. The Decision Tree is more precise but misses more. The ensemble sits between them.
- Combining models helped: the ensemble beat each individual model on accuracy.

## Limitations

- Oversampling by duplicating rows can make a model memorize copies of the minority class. Methods such as SMOTE, class weights, or tuned decision thresholds could be compared.
- The dataset is a 1994 US census sample, so the results say little about current income patterns, and the features include sensitive attributes such as race and sex.

## Data

[UCI Adult / Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult) (Becker & Kohavi, 1996).
