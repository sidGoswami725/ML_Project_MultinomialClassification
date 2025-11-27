import pandas as pd
from sklearn.model_selection import train_test_split

# --- File Paths ---
TRAIN_CLEANED_02 = "train_cleaned_02.csv"
TEST_CLEANED_02 = "test_cleaned_02.csv"

# Load the fully cleaned data
train_df = pd.read_csv(TRAIN_CLEANED_02)
test_df = pd.read_csv(TEST_CLEANED_02)

# Store trip_id and target (Y)
train_ids = train_df['trip_id']
test_ids = test_df['trip_id']
y_train = train_df['spend_category']

# Drop non-feature and target columns before combining for encoding
train_df.drop(columns=['trip_id', 'spend_category'], inplace=True)
test_df.drop(columns=['trip_id'], inplace=True)

# Combine for consistent encoding
combined_df = pd.concat([train_df, test_df], ignore_index=True)


# --- 1. Ordinal Encoding ---

# Define Ordinal Mappings (including 'Missing' as 0, the lowest rank)
age_mapping = {'Missing': 0, '15-24': 1, '25-44': 2, '45-64': 3, '65+': 4}
total_days_mapping = {'Missing': 0, '1-3': 1, '4-6': 2, '7-14': 3, '15-30': 4, '30+': 5}
booked_days_mapping = {'Missing': 0, '0-7': 1, '8-14': 2, '15-30': 3, '31-60': 4, '61-90': 5, '90+': 6}

# Apply the Mappings
combined_df['age_group'] = combined_df['age_group'].map(age_mapping)
combined_df['total_trip_days'] = combined_df['total_trip_days'].map(total_days_mapping)
combined_df['days_booked_before_trip'] = combined_df['days_booked_before_trip'].map(booked_days_mapping)


# --- 2. One-Hot Encoding (Nominal Categorical Features) ---

# Identify remaining object columns
nominal_cols = combined_df.select_dtypes(include='object').columns.tolist()

# Perform One-Hot Encoding (drop_first=True to avoid multicollinearity)
combined_df_encoded = pd.get_dummies(combined_df, columns=nominal_cols, drop_first=True, dtype=int)


# --- 3. Re-split into Train and Test Sets ---
X_train_encoded = combined_df_encoded.iloc[:len(train_df)].copy()
X_test_encoded = combined_df_encoded.iloc[len(train_df):].copy()

# Add trip_id and target back to the training set for final saving
X_train_encoded.insert(0, 'trip_id', train_ids.values)
X_test_encoded.insert(0, 'trip_id', test_ids.values)
X_train_encoded['spend_category'] = y_train.values


# --- Save the Final Preprocessed Files (pre-scaling) ---
X_train_encoded.to_csv("train_preprocessed_final.csv", index=False)
X_test_encoded.to_csv("test_preprocessed_final.csv", index=False)

print("feature_engineering_01_encoding.py executed successfully.")
print(f"Number of Features after encoding: {X_train_encoded.shape[1] - 2}")