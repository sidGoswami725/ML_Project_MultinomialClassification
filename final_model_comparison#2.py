import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "submission_fast_ensemble.csv"
SEED = 42

# --- 1. ADVANCED PREPROCESSING (Keep the good stuff) ---

def preprocess_data(train_path, test_path):
    print("Loading and preprocessing data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Drop rows with missing target in train
    train.dropna(subset=['spend_category'], inplace=True)
    train['spend_category'] = train['spend_category'].astype(int)
    
    test['spend_category'] = -1 
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # --- Cleanup ---
    df['num_females'] = df['num_females'].fillna(0)
    df['num_males'] = df['num_males'].fillna(0)
    df['mainland_stay_nights'] = df['mainland_stay_nights'].fillna(0)
    df['island_stay_nights'] = df['island_stay_nights'].fillna(0)
    
    # --- Feature Engineering ---
    
    # 1. FIX: Convert 'is_first_visit' to numeric
    df['is_first_visit'] = df['is_first_visit'].fillna('No').map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).astype(int)

    # 2. Wealth Signals
    df['total_people'] = df['num_females'] + df['num_males']
    df['is_alone'] = (df['total_people'] == 1).astype(int)
    
    binary_cols = [
        'intl_transport_included', 'accomodation_included', 'food_included',
        'domestic_transport_included', 'sightseeing_included', 
        'guide_included', 'insurance_included'
    ]
    for col in binary_cols:
        df[col] = df[col].fillna('No').map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    df['services_count'] = df[binary_cols].sum(axis=1)
    
    # 3. Ratios
    df['total_nights_calc'] = df['mainland_stay_nights'] + df['island_stay_nights']
    df['nights_per_person'] = df['total_nights_calc'] / (df['total_people'].replace(0, 1))
    
    # 4. Frequency Encoding (Crucial for Country)
    country_counts = df['country'].value_counts().to_dict()
    df['country_freq'] = df['country'].map(country_counts)
    
    # --- Encoding ---
    categorical_cols = [
        'country', 'age_group', 'travel_companions', 'main_activity', 
        'visit_purpose', 'tour_type', 'info_source', 'arrival_weather',
        'days_booked_before_trip', 'total_trip_days', 'has_special_requirements'
    ]
    
    for col in categorical_cols:
        df[col] = df[col].fillna('Missing')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # --- Final Split ---
    df_processed = df.drop(columns=['trip_id'])
    train_proc = df_processed[df_processed['spend_category'] != -1].copy()
    test_proc = df_processed[df_processed['spend_category'] == -1].copy()
    test_proc = test_proc.drop(columns=['spend_category'])
    
    return train_proc, test_proc, df[df['spend_category'] == -1]['trip_id']

# --- EXECUTION ---
train_df, test_df, test_ids = preprocess_data(TRAIN_FILE, TEST_FILE)

X = train_df.drop(columns=['spend_category'])
y = train_df['spend_category']

print(f"Data Shape: {X.shape}")

# --- 2. DEFINE OPTIMIZED MODELS (Faster settings) ---

# XGBoost
clf_xgb = XGBClassifier(
    n_estimators=1000,     # Reduced from 1500 for speed
    learning_rate=0.05,    # Increased slightly to compensate
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    random_state=SEED,
    n_jobs=-1
)

# LightGBM
clf_lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight='balanced',
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

# CatBoost
clf_cat = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    auto_class_weights='Balanced',
    random_seed=SEED,
    verbose=0,
    allow_writing_files=False
)

# --- 3. SOFT VOTING ENSEMBLE (Much Faster than Stacking) ---

print("\nTraining Soft Voting Ensemble...")
print("This trains each model ONCE (instead of 6 times for Stacking).")

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', clf_xgb),
        ('lgbm', clf_lgbm),
        ('cat', clf_cat)
    ],
    voting='soft', # Uses probabilities to vote
    n_jobs=-1
)

voting_clf.fit(X, y)

# --- 4. PREDICTION & SUBMISSION ---

print("Generating predictions...")
test_pred = voting_clf.predict(test_df)

submission = pd.DataFrame({
    'trip_id': test_ids,
    'spend_category': test_pred
})

submission.to_csv(SUBMISSION_FILE, index=False)

print(f"\n--- FAST ENSEMBLE SUBMISSION GENERATED ---")
print(f"File saved as: {SUBMISSION_FILE}")
print("Sample:")
print(submission.head())