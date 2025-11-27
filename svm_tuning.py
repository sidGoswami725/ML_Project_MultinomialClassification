import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss
import time

# --- File Paths ---
TRAIN_SCALED = "train_scaled_final.csv"

# --- Load Data ---
train_df = pd.read_csv(TRAIN_SCALED)
X = train_df.drop(columns=['trip_id', 'spend_category'])
y = train_df['spend_category'].values

# Final Imputation Fix (Ensure no NaNs before passing to model)
X.fillna(0, inplace=True)
X = X.values

# --- Setup ---
# Use a small number of splits (e.g., 3) for Grid Search due to SVC's high computational cost
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 

# Custom Log Loss Scorer for GridSearchCV
def neg_log_loss_scorer(estimator, X, y):
    """Custom scoring function for multi-class Log Loss (Cross-Entropy)."""
    try:
        # Predict probabilities (required for Log Loss)
        y_pred_proba = estimator.predict_proba(X)
        return -log_loss(y, y_pred_proba) # GridSearchCV minimizes the score
    except AttributeError:
        # Fallback if probability=True was somehow missed or failed
        print("Warning: Estimator does not support predict_proba, returning a high penalty.")
        return -100.0 

# --- Hyperparameter Grid Search Setup ---
param_grid = {
    # C: Regularization parameter. Smaller C promotes smoother decision boundary.
    'C': [0.1, 1.0, 10.0],
    # gamma: Kernel coefficient for 'rbf'. Controls the influence of individual training examples.
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf']
}

# Initialize Model
svm_model = SVC(
    probability=True, # CRITICAL: MUST be True for Log Loss calculation
    class_weight='balanced', # CRITICAL: Handles class imbalance
    random_state=42
)

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=svm_model,
    param_grid=param_grid,
    scoring=neg_log_loss_scorer,
    cv=skf,
    verbose=2, # Verbose output for monitoring progress
    n_jobs=-1 # Use all available cores for parallel processing
)

print("\nStarting SVM Grid Search (This will be computationally intensive and may take a while)...")
start_time = time.time()
# Fit the Grid Search
grid_search.fit(X, y)
end_time = time.time()
fit_time = end_time - start_time


# --- Report Results ---
# The best score from GridSearchCV is the negative log loss, so we negate it back.
best_log_loss = -grid_search.best_score_
best_params = grid_search.best_params_

print("\n--- Support Vector Machine (SVC) Tuning Results ---")
print(f"Total Fitting Time: {fit_time:.2f} seconds")
print(f"Best Log Loss (CV Mean): {best_log_loss:.4f}")
print(f"Best Hyperparameters: {best_params}")
print("\nModel execution complete.")