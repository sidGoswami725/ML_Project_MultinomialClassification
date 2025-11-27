import pandas as pd
import numpy as np
# Switched from ComplementNB to GaussianNB to handle Standard Scaled (continuous) data
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight

# --- File Paths ---
TRAIN_SCALED = "train_scaled_final.csv"

# --- Load Data ---
train_df = pd.read_csv(TRAIN_SCALED)

X = train_df.drop(columns=['trip_id', 'spend_category'])
y = train_df['spend_category'].values

# --- CRITICAL FIX: Final Imputation ---
# Check for remaining NaNs and impute with 0 immediately before model use.
nan_cols = X.columns[X.isnull().any()].tolist()
if nan_cols:
    print(f"Final CRITICAL FIX: Found {len(nan_cols)} columns with residual NaNs. Imputing with 0.")
    # Impute remaining NaNs with 0. This resolves the previous ValueErrors.
    X.fillna(0, inplace=True) 
else:
    print("Final CRITICAL FIX: Data confirmed clean (no NaNs).")

X = X.values # Convert to numpy array for model training

# --- 1. Handle Class Imbalance ---
classes = np.unique(y)
# Compute weights for reference, though GNB doesn't use the class_weight parameter directly
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))
print(f"Computed Class Weights: {class_weights}")


# --- 2. Initialize Model and Evaluation Setup ---
model = GaussianNB()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom scoring function for multi-class Log Loss
def log_loss_scorer(estimator, X, y):
    """Custom scoring function for multi-class Log Loss (Cross-Entropy)."""
    # The Log Loss requires probabilities
    y_pred_proba = estimator.predict_proba(X)
    # We return negative log loss because cross_val_score minimizes the score
    return -log_loss(y, y_pred_proba)


# --- 3. Evaluate using Log Loss (Cross-Validation) ---
print("\nStarting 5-Fold Cross-Validation for Gaussian Naive Bayes...")
cv_scores = cross_val_score(model, X, y, cv=skf, scoring=log_loss_scorer)

# Convert negative scores back to positive Log Loss
mean_log_loss = -cv_scores.mean()
std_log_loss = cv_scores.std()

# --- 4. Report Results ---
print("\n--- Gaussian Naive Bayes Baseline Results ---")
print(f"Log Loss (Cross-Validation Mean): {mean_log_loss:.4f}")
print(f"Log Loss (Cross-Validation Std Dev): {std_log_loss:.4f}")
print("\nModel execution complete.")