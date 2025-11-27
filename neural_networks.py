import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- File Paths ---
TRAIN_SCALED = "train_scaled_final.csv"

# --- Load Data ---
train_df = pd.read_csv(TRAIN_SCALED)
X = train_df.drop(columns=['trip_id', 'spend_category'])
y = train_df['spend_category'].values

# Final Imputation Fix (Ensure no NaNs)
X.fillna(0, inplace=True)
X = X.values

# Convert target variable to one-hot encoding (required for categorical_crossentropy)
Y_one_hot = to_categorical(y)
NUM_FEATURES = X.shape[1]
NUM_CLASSES = Y_one_hot.shape[1]

# --- 1. Calculate Class Weights (for Imbalance Handling) ---
classes = np.unique(y)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = dict(zip(classes, weights))
print(f"Computed Class Weights: {class_weights}")


# --- 2. Define the Neural Network Architecture ---
def create_model():
    model = Sequential([
        # Input Layer + Hidden Layer 1: Larger layer to handle the high dimensionality from OHE
        Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
        Dropout(0.3),

        # Hidden Layer 2: Smaller for feature compression
        Dense(64, activation='relu'),
        Dropout(0.3),

        # Output Layer: 3 units for 3 classes, 'softmax' activation for probabilities
        Dense(NUM_CLASSES, activation='softmax')
    ])

    # Compile the model using Adam optimizer and Log Loss (categorical_crossentropy)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model


# --- 3. Evaluate using Manual Stratified K-Fold CV ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_loss_scores = []
EPOCHS = 20 # Small number for initial fast assessment
BATCH_SIZE = 32

print("\nStarting Neural Network 5-Fold Cross-Validation...")

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/5 ---")
    X_train, X_val = X[train_index], X[val_index]
    Y_train_oh, Y_val_oh = Y_one_hot[train_index], Y_one_hot[val_index]

    # Re-create model for each fold to ensure fresh weights
    model = create_model()

    # Train the model using class weights
    model.fit(X_train, Y_train_oh,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=0, # Suppress verbose output for clean execution
              class_weight=class_weights) # Apply class weights

    # Predict probabilities on the validation set
    y_pred_proba = model.predict(X_val, verbose=0)
    
    # Calculate Log Loss
    fold_log_loss = log_loss(Y_val_oh, y_pred_proba)
    log_loss_scores.append(fold_log_loss)
    print(f"Fold {fold+1} Log Loss: {fold_log_loss:.4f}")


# --- 4. Report Results ---
mean_log_loss = np.mean(log_loss_scores)
std_log_loss = np.std(log_loss_scores)

print("\n--- Neural Network (MLP) Tuning Results ---")
print(f"Log Loss (Cross-Validation Mean): {mean_log_loss:.4f}")
print(f"Log Loss (Cross-Validation Std Dev): {std_log_loss:.4f}")
print("\nModel execution complete.")