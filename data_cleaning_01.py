import pandas as pd

# --- File Paths ---
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# Load the datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# --- 1. Drop rows with missing target variable from the training set ---
train_df.dropna(subset=['spend_category'], inplace=True)
train_df['spend_category'] = train_df['spend_category'].astype(int)

# --- 2. Binary Feature Encoding ('Yes'/'No' to 1/0) ---
binary_cols = [
    'is_first_visit',
    'intl_transport_included',
    'accomodation_included',
    'food_included',
    'domestic_transport_included',
    'sightseeing_included',
    'guide_included',
    'insurance_included'
]

def encode_binary(df, columns):
    """Maps 'Yes'/'No' to 1/0 and handles NaNs by mapping them to 0."""
    for col in columns:
        # Fill NaN with 0 for 'No', then map 'Yes' to 1
        df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
    return df

# Apply encoding to both train and test sets
train_df = encode_binary(train_df.copy(), binary_cols)
test_df = encode_binary(test_df.copy(), binary_cols)

# --- Save intermediate cleaned files ---
train_df.to_csv("train_cleaned_01.csv", index=False)
test_df.to_csv("test_cleaned_01.csv", index=False)

print("data_cleaning_01.py executed successfully.")
print(f"Cleaned Train size: {train_df.shape[0]} rows")