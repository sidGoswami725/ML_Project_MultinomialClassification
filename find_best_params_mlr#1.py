import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
import time

# --- File Paths ---
TRAIN_SCALED = "train_scaled_final.csv"

# --- Load Data ---
train_df = pd.read_csv(TRAIN_SCALED)
X = train_df.drop(columns=['trip_id', 'spend_category'])
y = train_df['spend_category'].values

# Final Imputation Fix
X.fillna(0, inplace=True)
X = X.values

# --- Setup ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 

# --- Hyperparameter Grid Search Setup ---
param_grid = {
    # C (Inverse of regularization strength): Smaller C = Stronger regularization
    'C': [0.1, 0.5, 1.0, 5.0, 10.0], 
    # Try both L1 (feature selection) and L2 (standard smoothing) penalties
    'penalty': ['l1', 'l2'] 
}

# Initialize Model
# 'saga' solver supports both L1 and L2 for multinomial logistic regression
mlr_model = LogisticRegression(
    multi_class='multinomial',
    solver='saga',
    max_iter=5000, # Increased iterations for convergence
    class_weight='balanced', # Crucial for handling class imbalance
    random_state=42
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=mlr_model,
    param_grid=param_grid,
    scoring='f1_weighted', # Optimize for Weighted F1-Score
    cv=skf,
    verbose=2,
    n_jobs=-1 # Use all available cores
)

print("\nStarting Multinomial Logistic Regression (MLR) Tuning...")
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()
fit_time = end_time - start_time


# --- Report Results ---
best_f1_score = grid_search.best_score_
best_params = grid_search.best_params_

print("\n--- Multinomial Logistic Regression Tuning Results ---")
print(f"Total Fitting Time: {fit_time:.2f} seconds")
print(f"Best Weighted F1-Score (CV Mean): {best_f1_score:.4f}")
print(f"Best Hyperparameters: {best_params}")
print("\nModel execution complete. Let's see if this linear model beats the SVM!")