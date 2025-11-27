import pandas as pd
import numpy as np
from sklearn.svm import SVC
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
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 

# --- Hyperparameter Grid Search Setup ---
param_grid = {
    # CRITICAL CHANGE: Focus on LOWER C values (less complex boundary = less overfitting)
    'C': [0.2, 0.5, 0.8, 1.0], 
    # Tighter gamma search around previous optimal (0.1)
    'gamma': [0.05, 0.1, 0.2], 
    'kernel': ['rbf']
}

# Initialize Model
svm_model = SVC(
    probability=False, 
    class_weight='balanced', 
    random_state=42
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring='f1_weighted', 
    cv=skf,
    verbose=2,
    n_jobs=-1
)

print("\nStarting REGULARIZED SVM Grid Search (Focusing on C < 1.0 to reduce overfitting)...")
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()
fit_time = end_time - start_time


# --- Report Results ---
best_f1_score = grid_search.best_score_
best_params = grid_search.best_params_

print("\n--- Regularized SVC Tuning Results ---")
print(f"Total Fitting Time: {fit_time:.2f} seconds")
print(f"Best Weighted F1-Score (CV Mean): {best_f1_score:.4f}")
print(f"Best Hyperparameters: {best_params}")
print("\nModel execution complete.")