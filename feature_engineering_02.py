import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- File Paths ---
TRAIN_PREPROCESSED = "train_preprocessed_final.csv"
TEST_PREPROCESSED = "test_preprocessed_final.csv"

# Load the fully encoded data
train_df = pd.read_csv(TRAIN_PREPROCESSED)
test_df = pd.read_csv(TEST_PREPROCESSED)

# Separate features (X) and target (y)
X_train = train_df.drop(columns=['trip_id', 'spend_category'])
y_train = train_df['spend_category']
X_test = test_df.drop(columns=['trip_id'])

# Identify the numerical/ordinal columns that need scaling
numerical_cols = [
    'num_females',
    'num_males',
    'mainland_stay_nights',
    'island_stay_nights',
    'age_group',
    'total_trip_days',
    'days_booked_before_trip'
]

# --- Apply Standard Scaling ---
scaler = StandardScaler()

# 1. Fit the scaler ONLY on the training data to prevent data leakage
X_train.loc[:, numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# 2. Transform the test data using the fitted scaler
X_test.loc[:, numerical_cols] = scaler.transform(X_test[numerical_cols])


# --- Final Assembly and Saving ---

X_train_final = X_train
X_test_final = X_test

# Add trip_id and target back for final saving
X_train_final['spend_category'] = y_train.values
X_train_final.insert(0, 'trip_id', train_df['trip_id'])
X_test_final.insert(0, 'trip_id', test_df['trip_id'])

# Save the final preprocessed files
X_train_final.to_csv("train_scaled_final.csv", index=False)
X_test_final.to_csv("test_scaled_final.csv", index=False)

print("feature_engineering_02_scaling.py executed successfully.")
print("Saved final scaled files: train_scaled_final.csv and test_scaled_final.csv")