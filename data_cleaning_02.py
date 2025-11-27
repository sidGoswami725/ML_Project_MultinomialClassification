import pandas as pd

# --- File Paths ---
TRAIN_CLEANED_01 = "train_cleaned_01.csv"
TEST_CLEANED_01 = "test_cleaned_01.csv"

# Load the intermediate cleaned data
train_df = pd.read_csv(TRAIN_CLEANED_01)
test_df = pd.read_csv(TEST_CLEANED_01)

# Columns to fill with 'Missing' category (high proportion of NaNs)
fill_missing_category = ['has_special_requirements', 'arrival_weather', 'days_booked_before_trip']

# Columns to fill with Mode (low proportion of NaNs)
fill_mode = ['country', 'age_group', 'travel_companions', 'main_activity', 'total_trip_days', 'num_females', 'num_males']

def impute_data(df, fill_missing, fill_mode_cols, train_data):
    """Handles imputation on a given DataFrame."""
    # 1. Fill high NaN columns with 'Missing'
    for col in fill_missing:
        if col in df.columns:
            df[col].fillna('Missing', inplace=True)

    # 2. Fill low NaN columns with Mode (using mode calculated ONLY from training data)
    for col in fill_mode_cols:
        if col in train_data.columns and col in df.columns:
            # Calculate mode only from the training set to prevent data leakage
            mode_val = train_data[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            
    return df

# Apply imputation
train_df = impute_data(train_df, fill_missing_category, fill_mode, train_df.copy())
test_df = impute_data(test_df, fill_missing_category, fill_mode, train_df.copy())

# Convert num_females and num_males back to integer type
train_df['num_females'] = train_df['num_females'].astype(int)
train_df['num_males'] = train_df['num_males'].astype(int)
test_df['num_females'] = test_df['num_females'].astype(int)
test_df['num_males'] = test_df['num_males'].astype(int)

# --- Save the fully cleaned files ---
train_df.to_csv("train_cleaned_02.csv", index=False)
test_df.to_csv("test_cleaned_02.csv", index=False)

print("data_cleaning_02.py executed successfully. All missing values handled.")