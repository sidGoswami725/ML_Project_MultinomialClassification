import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# Required models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC # Using SVC for RBF kernel
from sklearn.naive_bayes import GaussianNB # Using GaussianNB for Bayes

# Optional performance models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False
    print("XGBoost not installed. Skipping.")


# --- 1) LOAD AND PREPARE DATA (Using Pre-processed Files) ---
# We use the final scaled files from Phase III, skipping initial upload/raw preprocessing.
TRAIN_FILE = "train_scaled_final.csv"
TEST_FILE = "test_scaled_final.csv"

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate X and y
X = train_df.drop(columns=['trip_id', 'spend_category'])
y = train_df['spend_category']

test_ids = test_df["trip_id"]
X_test = test_df.drop(columns=['trip_id'])

# CRITICAL FIX: Final Imputation (Safety Check from previous runs)
X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print(f"X (Train Features): {X.shape} | X_test (Test Features): {X_test.shape}")


# --- 2) TRAIN/VALIDATION SPLIT (For quick internal scoring) ---
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# --- 3) DEFINE MODELS (Curated for Multi-Class & Imbalance) ---
# Note: StandardScaler is removed from pipelines as data is already scaled.
# All models use class_weight='balanced' for the imbalanced multi-class problem.

models = {}

# 1. Multinomial Logistic Regression (MLR - Strong Regularization)
models["MLR"] = LogisticRegression(
    multi_class='multinomial', 
    solver='saga', 
    max_iter=5000, 
    C=1.0, # Default C for quick test
    class_weight='balanced', 
    random_state=42
)

# 2. Random Forest Classifier
models["RandomForest"] = RandomForestClassifier(
    n_estimators=300, 
    max_depth=12, 
    class_weight='balanced',
    random_state=42
)

# 3. Gradient Boosting Classifier
models["GradientBoosting"] = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

# 4. Neural Network (MLP - Required model)
models["NN"] = MLPClassifier(
    hidden_layer_sizes=(128, 64), 
    activation="relu", 
    solver="adam", 
    max_iter=300, # Increased max_iter for convergence
    random_state=42
)

# 5. Gaussian Naive Bayes (Required model)
models["GaussianNB"] = GaussianNB()

# 6. Support Vector Machine (SVC - Our best model so far)
models["SVM_Optimized"] = SVC(
    C=1.0, 
    gamma=0.2, # Best parameter from regularization tuning
    kernel='rbf', 
    class_weight='balanced',
    random_state=42
)

if HAS_XGB:
    # 7. XGBoost Classifier
    models["XGBoost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax', # Multi-class objective
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )

print("\nDefined 6-7 Models for comparison.")


# --- 4) TRAIN MODELS + VALIDATE + SAVE SUBMISSIONS ---

f1_scores = {}
submission_files = {}

for name, model in models.items():
    print(f"\n🚀 Training {name} ...")
    
    # --- TRAIN on Validation Split ---
    model.fit(X_train.values, y_train.values)

    # --- VALIDATION F1-Score ---
    val_pred = model.predict(X_valid.values)
    # Using weighted F1-score for multi-class and imbalance
    f1 = f1_score(y_valid.values, val_pred, average='weighted')
    f1_scores[name] = f1
    print(f"{name} Weighted F1-Score (Validation) = {f1:.4f}")

    # --- TRAIN on FULL training set ---
    model.fit(X.values, y.values)

    # --- Test predictions (Hard labels 0, 1, 2) ---
    test_pred = model.predict(X_test.values).astype(int)

    # --------------------------
    # SAVE SUBMISSION FILE (Two-column format)
    # --------------------------
    submission_df = pd.DataFrame({
        "trip_id": test_ids,
        "spend_category": test_pred
    })

    filename = f"submission_{name.lower()}.csv"
    submission_df.to_csv(filename, index=False)
    submission_files[name] = filename
    print(f"✔ Saved submission: {filename}")


# --- 5) FINAL SUMMARY ---
print("\n=======================================================")
print("✅ FINAL COMPARISON (Validation F1-Score):")
best_model_name = max(f1_scores, key=f1_scores.get)
for name, f1 in sorted(f1_scores.items(), key=lambda item: item[1], reverse=True):
    print(f"  {name:20s} -> F1 Score: {f1:.4f}")

print(f"\n🥇 The best internal model is: {best_model_name}")
print(f"➡ Submit {submission_files[best_model_name]} to Kaggle.")
print("=======================================================")