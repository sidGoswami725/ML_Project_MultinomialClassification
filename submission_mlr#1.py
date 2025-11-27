import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- File Paths ---
TRAIN_SCALED = "train_scaled_final.csv"
TEST_SCALED = "test_scaled_final.csv"
SUBMISSION_FILE = "submission_mlr_default.csv" # New file name

# --- Load Data ---
train_df = pd.read_csv(TRAIN_SCALED)
test_df = pd.read_csv(TEST_SCALED)

# Separate Features (X) and Target (y)
X_train = train_df.drop(columns=['trip_id', 'spend_category'])
y_train = train_df['spend_category'].values
X_test = test_df.drop(columns=['trip_id'])
test_ids = test_df['trip_id']

# --- Final Imputation Fix (Safety Check) ---
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

X_train = X_train.values
X_test = X_test.values

# --- 1. Define the DEFAULT MLR Model ---
default_mlr_model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs', # Default solver, suitable for multinomial classification
    penalty='l2', # Default penalty
    C=1.0, # Default regularization strength
    max_iter=5000, # Increased iterations for convergence safety
    class_weight='balanced', # CRUCIAL for our imbalanced data
    random_state=42
)

print("Training default Multinomial LR model on full dataset...")
# --- 2. Train Model on Full Training Data ---
default_mlr_model.fit(X_train, y_train)

# --- 3. Generate Predictions on Test Data ---
print("Generating class label predictions for the test set...")
# predict returns the predicted class label (0, 1, or 2)
test_predictions_label = default_mlr_model.predict(X_test)

# --- 4. Create Submission File (2 Columns) ---
submission_df = pd.DataFrame({
    'trip_id': test_ids,
    # The column name must match the sample file
    'spend_category': test_predictions_label 
})

# Save the final submission file
submission_df.to_csv(SUBMISSION_FILE, index=False)

print("\n--- MLR Default Submission File Creation Complete ---")
print(f"File saved as: {SUBMISSION_FILE}")
print("Submission file head:")
print(submission_df.head())
print("\nPlease submit this file to Kaggle.")