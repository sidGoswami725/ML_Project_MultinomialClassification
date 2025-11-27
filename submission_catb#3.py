import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "submission_catboost_pro_cv.csv"
SEED = 42
N_FOLDS = 5  # K-Fold Cross Validation

# --- 1. DATA PREPROCESSING ---

def preprocess_data(train_path, test_path):
    print("Loading and preprocessing data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Drop rows with missing target in train
    train.dropna(subset=['spend_category'], inplace=True)
    train['spend_category'] = train['spend_category'].astype(int)

    test['spend_category'] = -1
    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    # --- Feature Engineering ---

    # 1. Fix 'is_first_visit' (Critical for CatBoost)
    df['is_first_visit'] = df['is_first_visit'].fillna('No').map({'Yes': '1', 'No': '0', 'yes': '1', 'no': '0'})

    # 2. Impute Counts
    df['num_females'] = df['num_females'].fillna(0)
    df['num_males'] = df['num_males'].fillna(0)
    df['total_people'] = df['num_females'] + df['num_males']

    # 3. Services Count (Wealth Signal)
    binary_cols = [
        'intl_transport_included', 'accomodation_included', 'food_included',
        'domestic_transport_included', 'sightseeing_included',
        'guide_included', 'insurance_included'
    ]
    # Fill with "No" string for now
    for col in binary_cols:
        df[col] = df[col].fillna('No').astype(str)

    # Create a numeric counter for feature engineering
    temp_services = df[binary_cols].replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
    df['services_count'] = temp_services.sum(axis=1)

    # 4. Ratios (Cost per person logic)
    df['mainland_stay_nights'] = df['mainland_stay_nights'].fillna(0)
    df['island_stay_nights'] = df['island_stay_nights'].fillna(0)
    df['total_nights'] = df['mainland_stay_nights'] + df['island_stay_nights']

    # Avoid division by zero
    df['nights_per_person'] = df['total_nights'] / (df['total_people'].replace(0, 1))

    # 5. Country Frequency (Is this a rare destination?)
    # High-spend travelers often go to rarer destinations
    country_counts = df['country'].value_counts().to_dict()
    df['country_freq'] = df['country'].map(country_counts)

    # --- CATEGORICAL SETUP ---
    # Define which columns are categorical. CatBoost wants them as Strings or Integers.
    # We will convert all to strings to be safe.
    categorical_cols = [
        'country', 'age_group', 'travel_companions', 'main_activity',
        'visit_purpose', 'tour_type', 'info_source', 'arrival_weather',
        'days_booked_before_trip', 'total_trip_days', 'has_special_requirements',
        'is_first_visit'
    ] + binary_cols # Treat binary Yes/No as categories too

    for col in categorical_cols:
        df[col] = df[col].fillna('Missing').astype(str)

    # --- Final Split ---
    train_proc = df[df['spend_category'] != -1].copy()
    test_proc = df[df['spend_category'] == -1].copy()
    test_ids = test_proc['trip_id']

    # Drop IDs and Target from features
    X = train_proc.drop(columns=['trip_id', 'spend_category'])
    y = train_proc['spend_category']
    X_test = test_proc.drop(columns=['trip_id', 'spend_category'])

    return X, y, X_test, test_ids, categorical_cols

# --- EXECUTION ---
X, y, X_test, test_ids, cat_features = preprocess_data(TRAIN_FILE, TEST_FILE)

print(f"Data Shape: {X.shape}")
print(f"Categorical Features identified: {len(cat_features)}")


# --- 2. 5-FOLD CROSS-VALIDATION TRAINING ---
#

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Arrays to store predictions
# We store probabilities: shape (n_test_samples, 3_classes)
oof_preds = np.zeros((X.shape[0], 3))
test_probs_sum = np.zeros((X_test.shape[0], 3))

f1_scores = []

print(f"\nStarting {N_FOLDS}-Fold Cross-Validation with CatBoost...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize CatBoost
    clf = CatBoostClassifier(
        iterations=1500,        # Increased iterations
        learning_rate=0.03,     # Lower LR for better convergence
        depth=7,                # Slightly deeper trees
        loss_function='MultiClass',
        auto_class_weights='Balanced',
        cat_features=cat_features, # NATIVE HANDLING
        random_seed=SEED,
        verbose=0,
        allow_writing_files=False,
        early_stopping_rounds=50
    )

    # Fit with early stopping (needs eval_set)
    clf.fit(
        X_train_fold, y_train_fold,
        eval_set=(X_val_fold, y_val_fold),
        use_best_model=True,
        verbose=False
    )

    # Predict on Validation (for score check)
    val_pred = clf.predict(X_val_fold).flatten()
    f1 = f1_score(y_val_fold, val_pred, average='weighted')
    f1_scores.append(f1)

    # Predict Probabilities on Test Set
    test_probs_sum += clf.predict_proba(X_test)

    print(f"Fold {fold+1} Weighted F1: {f1:.4f}")

# --- 3. AGGREGATION & SUBMISSION ---

avg_f1 = np.mean(f1_scores)
print(f"\nAverage CV Weighted F1-Score: {avg_f1:.4f}")

# Average the test probabilities across all 5 folds
test_probs_avg = test_probs_sum / N_FOLDS

# Take the class with highest probability (Argmax)
final_predictions = np.argmax(test_probs_avg, axis=1)

submission = pd.DataFrame({
    'trip_id': test_ids,
    'spend_category': final_predictions
})

submission.to_csv(SUBMISSION_FILE, index=False)

print(f"\n--- CATBOOST PRO SUBMISSION GENERATED ---")
print(f"File saved as: {SUBMISSION_FILE}")
print("Sample:")
print(submission.head())