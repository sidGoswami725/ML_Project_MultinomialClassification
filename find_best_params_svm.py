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

# Final Imputation Fix (Ensure no NaNs)
X.fillna(0, inplace=True)
X = X.values

# --- Setup ---
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 

# --- Hyperparameter Grid Search Setup ---
# Expanded search space for better optimization
param_grid = {
    # Wider range for C: 0.5 is between the previous C=0.1 and C=1.0
    'C': [0.5, 1.0, 5.0, 10.0], 
    # Expanded gamma: checking smaller (0.005) and larger (0.5) values
    'gamma': [0.005, 0.01, 0.1, 0.5], 
    'kernel': ['rbf']
}

# Initialize Model
# probability=False is faster and sufficient for F1-score tuning
svm_model = SVC(
    probability=False, 
    class_weight='balanced', # Still essential for handling imbalance
    random_state=42
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    # CRITICAL CHANGE: Optimize for Weighted F1-Score (higher is better)
    scoring='f1_weighted', 
    cv=skf,
    verbose=2,
    n_jobs=-1 # Use all available cores
)

print("\nStarting DEEPER SVM Grid Search (Optimizing for Weighted F1-Score)...")
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()
fit_time = end_time - start_time


# --- Report Results ---
best_f1_score = grid_search.best_score_
best_params = grid_search.best_params_

print("\n--- Support Vector Machine (SVC) Deep Tuning Results ---")
print(f"Total Fitting Time: {fit_time:.2f} seconds")
print(f"Best Weighted F1-Score (CV Mean): {best_f1_score:.4f}")
print(f"Best Hyperparameters: {best_params}")
print("\nModel execution complete. Ready to generate improved submission.")