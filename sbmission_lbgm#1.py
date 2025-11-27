import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier # Switched to LightGBM
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "submission_lgbm_final.csv"
SEED = 42

# --- 1. ADVANCED PREPROCESSING & FEATURE ENGINEERING (Identical) ---

def preprocess_data(train_path, test_path):
    """Loads, cleans, engineers features, and label-encodes categorical data."""
    print("Loading and preprocessing data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Drop rows with missing target in train
    train.dropna(subset=['spend_category'], inplace=True)
    train['spend_category'] = train['spend_category'].astype(int)
    
    # Combine for consistent engineering
    test['spend_category'] = -1 # Placeholder
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- A. Feature Engineering ---
    
    # 1. Total People
    df['num_females'] = df['num_females'].fillna(0)
    df['num_males'] = df['num_males'].fillna(0)
    df['total_people'] = df['num_females'] + df['num_males']
    
    # 2. Services Count 
    binary_cols = [
        'intl_transport_included', 'accomodation_included', 'food_included',
        'domestic_transport_included', 'sightseeing_included', 
        'guide_included', 'insurance_included'
    ]
    
    for col in binary_cols:
        df[col] = df[col].fillna('No').map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
        
    df['services_count'] = df[binary_cols].sum(axis=1)
    
    # 3. Stay Duration Logic
    df['mainland_stay_nights'] = df['mainland_stay_nights'].fillna(0)
    df['island_stay_nights'] = df['island_stay_nights'].fillna(0)
    df['total_nights_calc'] = df['mainland_stay_nights'] + df['island_stay_nights']
    
    # 4. Binary Flags
    df['is_alone'] = (df['total_people'] == 1).astype(int)
    df['is_first_visit'] = df['is_first_visit'].map({'Yes': 1, 'No': 0}).fillna(0)
    
    # --- B. Handling Categorical Data (Label Encoding) ---
    categorical_cols = [
        'country', 'age_group', 'travel_companions', 'main_activity', 
        'visit_purpose', 'tour_type', 'info_source', 'arrival_weather',
        'days_booked_before_trip', 'total_trip_days', 'has_special_requirements'
    ]
    
    for col in categorical_cols:
        df[col] = df[col].fillna('Missing')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # --- C. Final Cleanup ---
    df_processed = df.drop(columns=['trip_id', 'num_females', 'num_males', 'mainland_stay_nights', 'island_stay_nights'])
    
    # Split back into train and test
    train_proc = df_processed[df_processed['spend_category'] != -1].copy()
    test_proc = df_processed[df_processed['spend_category'] == -1].copy()
    test_ids = df[df['spend_category'] == -1]['trip_id']
    test_proc = test_proc.drop(columns=['spend_category'])
    
    return train_proc, test_proc, test_ids

# --- EXECUTION: DATA PREP ---
train_df, test_df, test_ids = preprocess_data(TRAIN_FILE, TEST_FILE)

X = train_df.drop(columns=['spend_category'])
y = train_df['spend_category']

print(f"Data Shape: {X.shape}")
print("Features engineered and data is ready for modeling.")

# --- 2. MODEL DEFINITION & TRAINING (LightGBM) ---

# Split for internal validation F1-Score
X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print("\nTraining LightGBM Classifier...")

# Initialize LightGBM
clf_lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced', # Built-in balancing
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

# Train on validation split to get F1 score
clf_lgbm.fit(X_train_split, y_train_split)

# Validation F1-Score
val_pred = clf_lgbm.predict(X_valid_split)
f1 = f1_score(y_valid_split, val_pred, average='weighted')
print(f"LightGBM Weighted F1-Score (Validation) = {f1:.4f}")

# Train on FULL training set for final submission
clf_lgbm.fit(X, y)

# --- 3. PREDICTION & SUBMISSION ---

print("Generating final predictions and submission file...")
test_pred = clf_lgbm.predict(test_df).astype(int)

submission = pd.DataFrame({
    'trip_id': test_ids,
    'spend_category': test_pred
})

submission.to_csv(SUBMISSION_FILE, index=False)

print(f"\n--- LightGBM SUBMISSION GENERATED ---")
print(f"File saved as: {SUBMISSION_FILE}")
print("Sample:")
print(submission.head())