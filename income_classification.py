import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def knn_imputation(file_path):
    """
    Data Cleaning & KNN Imputation
    This portion curates the data by filling in the missing values with 
    the most likely value based on KNN. It also loads the data into the data structure.
    """
    print(f"\n Starting KNN Imputation, oversampling, One-Hot Encoding with z-score scaling on {file_path}")
    
    COLUMN_NAMES = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    #Load Data
    df = pd.read_csv(
        file_path,
        header=None,
        names=COLUMN_NAMES,
        sep=r',\s*',
        engine='python', 
        na_values=['?', ' ?'],
        skipinitialspace=True
    )

    if 'income' in df.columns:
        df['income'] = df['income'].astype(str).str.rstrip('.')

    total_nans_raw = df.isnull().sum().sum()
    print(f"Total missing values found: {total_nans_raw}")

    #Target (Income)
    df_target = df[['income']]
    df_features = df.drop('income', axis=1)

    #Identify Column Types
    cat_cols = df_features.select_dtypes(include=['object']).columns
    num_cols = df_features.select_dtypes(include=['int64', 'float64']).columns

    # Create Pipeline for KNN Preparation (Encode Cats, Scale Nums)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # Fit and transform the data
    X_encoded = preprocessor.fit_transform(df_features)
    
    #Run KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_encoded)

    #Reconstruct DataFrame 
    df_clean = df.copy()
    
    # Extract just the numeric part of the imputed array (first len(num_cols) columns)
    X_imputed_num = X_imputed[:, :len(num_cols)]
    scaler = preprocessor.named_transformers_['num']
    X_restored_num = scaler.inverse_transform(X_imputed_num)
    
    df_clean[num_cols] = X_restored_num

    # For categorical columns, simple mode imputation 
    for col in cat_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    return df_clean

def oversampling(df):
    """
    Handle Class Imbalance using RandomOverSampler
    """
   
    X = df.drop('income', axis=1)
    y = df['income']
    
    print(f"Original Class Distribution:\n{y.value_counts()}")
    
    ros = RandomOverSampler(random_state=50)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    # Recombine
    balanced_df = pd.concat([X_resampled, y_resampled], axis=1)
    
    return balanced_df

def encoding_scaling(train_df, test_df=None):
    """
    Final Encoding and Scaling
    - Fits on train data
    - Transforms train data
    - If test data exists, Transforms Test data using train scalers
    """
   
    y_train = train_df['income']
    X_train = train_df.drop('income', axis=1)
    
    cat_cols = X_train.select_dtypes(include=['object']).columns
    if 'income' in cat_cols: cat_cols = cat_cols.drop('income')
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    X_train_encoded = preprocessor.fit_transform(X_train)
    
    new_column_names = list(num_cols) + list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    
    if hasattr(X_train_encoded, "toarray"): X_train_encoded = X_train_encoded.toarray()
    X_train_final = pd.DataFrame(X_train_encoded, columns=new_column_names)
    final_train_df = pd.concat([X_train_final, y_train.reset_index(drop=True)], axis=1)
    
    final_test_df = None
    
    if test_df is not None:
        print("Transforming Test Data (using Training scalers).")
        y_test = test_df['income']
        X_test = test_df.drop('income', axis=1)
        
        X_test_encoded = preprocessor.transform(X_test)
        
        if hasattr(X_test_encoded, "toarray"): X_test_encoded = X_test_encoded.toarray()
        X_test_final = pd.DataFrame(X_test_encoded, columns=new_column_names)
        final_test_df = pd.concat([X_test_final, y_test.reset_index(drop=True)], axis=1)

    return final_train_df, final_test_df

def decision_tree(train_df, test_df=None):
    """
    Method 1: Decision Tree
    """
    print("\n Method 1: Decision Tree Training & Evaluation ")

    X_train_full = train_df.drop('income', axis=1)
    y_train_full = train_df['income']

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_full)
    print(f"Target classes encoded: {le.classes_}")

    if test_df is not None:
        print("Using External Test File for evaluation.")
        X_train, y_train = X_train_full, y_train_enc
        
        # Prepare 
        X_test = test_df.drop('income', axis=1)
        y_test = le.transform(test_df['income']) 
    else:
        print("Using Internal 80/20 Split.")
        X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_enc, test_size=0.2, random_state=50)

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train
    clf = DecisionTreeClassifier(random_state=50)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    #Eval
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Decision Tree Complete.")
    return clf

def logistic_regression(train_df, test_df=None):
    """
    Method 2: Manual Logistic Regression (Gradient Descent)
    """
    print("\nMethod 2: Manual Logistic Regression (Gradient Descent)")
    
    # Prepare Train
    X_train_full = train_df.drop("income", axis=1)
    y_train_full = train_df["income"]

    # Map Target consistently
    unique_vals = y_train_full.unique()
    
    #Sort to ensure 0 is first alphabetically or consistently
    unique_vals.sort()
    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    print(f"Mapping target values: {mapping}")
    
    y_train_mapped = y_train_full.map(mapping)
    X_train_dummies = pd.get_dummies(X_train_full) 

    if test_df is not None:
        X_train = X_train_dummies.values
        y_train = y_train_mapped.values
        
        X_test_dummies = pd.get_dummies(test_df.drop("income", axis=1))
        X_test_dummies = X_test_dummies.reindex(columns=X_train_dummies.columns, fill_value=0)
        
        X_test = X_test_dummies.values
        y_test = test_df["income"].map(mapping).values
    else:
        split = int(0.8 * len(train_df))
        X_train = X_train_dummies[:split].values
        X_test = X_train_dummies[split:].values
        y_train = y_train_mapped[:split].values
        y_test = y_train_mapped[split:].values

    weights = np.zeros(X_train.shape[1])
    bias = 0
    learning_rate = 0.05
    iterations = 5000

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    print(f"Training Logistic Regression for {iterations} iterations.")
    for i in range(iterations):
        predictions = sigmoid(np.dot(X_train, weights) + bias)
        errors = predictions - y_train
        weights = weights - learning_rate * np.dot(X_train.T, errors) / len(y_train)
        bias = bias - learning_rate * np.sum(errors) / len(y_train)
        
        if i % 1000 == 0:
            print(f"Iteration {i}")

    test_predictions = sigmoid(np.dot(X_test, weights) + bias)
    test_predictions = (test_predictions > 0.5).astype(int)

    accuracy = np.mean(test_predictions == y_test)
    print(f"\nLogistic Regression Accuracy: {accuracy * 100:.2f}%")
    print("Logistics Complete.")
    
    #FOR REVIEWER: If you would like to generate the figures and plots, as displayed in our presentation and report - uncomment the following code and run as needed
    #these are all plots as presented in finalized work
    #Get probability predictions for visualization
    #test_predictions_proba = sigmoid(np.dot(X_test, weights) + bias)

    # Create figure with 2 subplots for model eval.
    #fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    #Sigmoid S-Curve
    #ax1 = axes[0]
    #z_range = np.linspace(-10, 10, 100)
    #sigmoid_values = sigmoid(z_range)
    #ax1.plot(z_range, sigmoid_values, 'b-', linewidth=2)
    #ax1.axhline(y=0.5, color='r', linestyle='--', label='Decision Boundary (0.5)')
    #ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    #ax1.grid(True, alpha=0.3)
    #ax1.set_xlabel('z (weighted sum)', fontsize=11)
    #ax1.set_ylabel('Probability', fontsize=11)
    #ax1.set_title('Sigmoid Function (S-Curve)', fontsize=12, fontweight='bold')
    #ax1.legend()

    #Model Performance Metrics (box)
    #ax2 = axes[1]
    #ax2.axis('off')

    #Calculate metrics
    #true_pos = np.sum((test_predictions == 1) & (y_test == 1))
    #true_neg = np.sum((test_predictions == 0) & (y_test == 0))
    #false_pos = np.sum((test_predictions == 1) & (y_test == 0))
    #false_neg = np.sum((test_predictions == 0) & (y_test == 1))

    #Calculate metrics
    #precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    #recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    #f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    #metrics_text = f"""
    #Model Performance Metrics
    
    #Accuracy: {accuracy * 100:.2f}%

    #Precision: {precision * 100:.2f}%

    #Recall: {recall * 100:.2f}%

    #F1-Score: {f1_score * 100:.2f}%

    #True Positives: {true_pos}
    #True Negatives: {true_neg}
    #False Positives: {false_pos}
    #False Negatives: {false_neg}
    

    #ax2.text(0.5, 0.5, metrics_text, ha='center', va='center', 
    #     fontsize=12, family='monospace',
    #    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    #plt.tight_layout()
    #plt.savefig('logistic_regression_visualizations.png', dpi=300, bbox_inches='tight')
    #plt.show()

def knn_classification(train_df, test_df=None):
    """
    Method 3: KNN Classification
    """
    print("\nMethod 3: KNN Classification")
    
    X_train_full = train_df.drop("income", axis=1)
    y_train_full = train_df["income"]

    if test_df is not None:
        X_train, y_train = X_train_full, y_train_full
        X_test = test_df.drop("income", axis=1)
        y_test = test_df["income"]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=50)

    print("Training KNN (n_neighbors=5)")
    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("KNN Complete.")
    #FOR REVIEWER: If you would like to generate the figures and plots, as displayed in our presentation and report - uncomment the following code and run as needed
    #these are all KNN plot as presented in finalized work

    #INCOME DISTRIBUTION
    # df["income"].value_counts().plot(kind="bar")
    # plt.title("Income Class Distribution")
    # plt.xlabel("Income Class")
    # plt.ylabel("Count")
    # plt.show()
    #
    #FEATURE DISTRIBUTION
    # df.hist(figsize=(12, 10))
    # plt.suptitle("Attribute Distributions")
    # plt.show()

    #CONFUSION MATRIX
    # from sklearn.metrics import ConfusionMatrixDisplay
    # display = ConfusionMatrixDisplay.from_predictions(
    #     y_test, y_pred, cmap="Blues")
    # plt.title("KNN Confusion Matrix")
    # plt.show()


    #ACCURACY V. K
    # k_values = range(1, 10)
    # accuracies = []
    #
    # for k in k_values:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knn.fit(X_train, y_train)
    #     y_pred_k = knn.predict(X_test)
    #     accuracies.append(accuracy_score(y_test, y_pred_k))
    #
    # plt.plot(k_values, accuracies)
    # plt.xlabel("Number of Neighbors")
    # plt.ylabel("Accuracy")
    # plt.title("Model Accuracy vs K")
    # plt.show()
    
def weighted_ensemble(train_df, test_df=None):
    """
    Weighted Ensemble Model
    """
    print("\nWeighted Ensemble (DT + Manual LR + KNN)")
    
    X = train_df.drop('income', axis=1)
    y = train_df['income']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    if test_df is not None:
        print("Ensemble: Using External Test Data.")
        X_train, y_train = X, y_encoded
        X_test = test_df.drop('income', axis=1)
        y_test = le.transform(test_df['income'])
    else:
        print("Ensemble: Using Internal Split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=50)
    
    print(f"Ensemble Training Set: {X_train.shape[0]}")
    print(f"Ensemble Test Set: {X_test.shape[0]}")

    #DT
    clf = DecisionTreeClassifier(random_state=50)
    clf.fit(X_train, y_train)
    prob_dt = clf.predict_proba(X_test)[:, 1]

    #KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn.fit(X_train, y_train)
    prob_knn = knn.predict_proba(X_test)[:, 1]

    #Manual LR
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train
    
    weights = np.zeros(X_train_np.shape[1])
    bias = 0
    learning_rate = 0.05
    iterations = 5000

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    for i in range(iterations):
        predictions = sigmoid(np.dot(X_train_np, weights) + bias)
        errors = predictions - y_train_np
        weights = weights - learning_rate * np.dot(X_train_np.T, errors) / len(y_train_np)
        bias = bias - learning_rate * np.sum(errors) / len(y_train_np)
    
    prob_lr = sigmoid(np.dot(X_test_np, weights) + bias)

    # Combine
    w_dt = 0.32
    w_lr = 0.33
    w_knn = 0.35
    
    print(f"Applying weights -> DT: {w_dt}, LR: {w_lr}, KNN: {w_knn}")
    weighted_prob = (prob_dt * w_dt + prob_lr * w_lr + prob_knn * w_knn) / (w_dt + w_lr + w_knn)
    final_preds = (weighted_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, final_preds)
    print(f"\nEnsemble Model Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, final_preds))
    print("Ensemble Complete.")


# --- Main Pipeline Execution ---
if __name__ == "__main__":
    TRAIN_INPUT_FILE = 'census-income.data.csv'
    
    # Set this to 'census-income.test.csv' (or your filename) to use external testing
    # Set to None to use the original 80/20 split on the training file
    #TEST_INPUT_FILE = None 
    TEST_INPUT_FILE = 'census-income.test.csv'

    FINAL_OUTPUT_FILE = 'final-train.csv'

    try:
        #Load & Preprocess TRAIN
        train_clean = knn_imputation(TRAIN_INPUT_FILE)
        train_balanced = oversampling(train_clean)
        
        #Load & Preprocess TEST
        test_processed = None
        if TEST_INPUT_FILE:
            print(f"\nProcessing External Test File: {TEST_INPUT_FILE}")
            test_clean = knn_imputation(TEST_INPUT_FILE)
            final_train_df, final_test_df = encoding_scaling(train_balanced, test_clean)
            test_processed = final_test_df
        else:
            final_train_df, _ = encoding_scaling(train_balanced, None)

        #Save Final Result - Training set
        final_train_df.to_csv(FINAL_OUTPUT_FILE, index=False)
        print(f"\nFinal training dataset saved to: {FINAL_OUTPUT_FILE}")
        print(f"\nTraining and Test (if provided) dataset have been curated. Oversampling not run on TEST: {FINAL_OUTPUT_FILE}")
        
        # Run Models
        # Pass the processed test df (if it exists) to all models
        decision_tree(final_train_df, test_processed)
        logistic_regression(final_train_df.copy(), test_processed) 
        knn_classification(final_train_df.copy(), test_processed)
        weighted_ensemble(final_train_df.copy(), test_processed)

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
