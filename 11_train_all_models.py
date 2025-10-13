"""
F1 Podium Predictor - Model Training & Comparison
Trains 4 models: Logistic Regression, Random Forest, XGBoost, Neural Network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("🏎️  F1 PODIUM PREDICTOR - MODEL TRAINING")
print("=" * 80)

# ============================================
# 1. LOAD AND PREPARE DATA
# ============================================
print("📥 Loading engineered features...")

df = pd.read_csv('f1_data_with_features.csv')
print(f"✅ Loaded {len(df)} entries with {len(df.columns)} features\n")

# Separate features and target
target = 'on_podium'
exclude_cols = [
    'year', 'race_name', 'race_date', 'circuit_name', 'country',
    'driver_abbreviation', 'driver_number', 'team_name',
    'finish_position', 'on_podium', 'points_scored'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y = df[target]

# Handle missing values
print(f"🔍 Checking for missing values...")
nan_counts = X.isna().sum()
nan_features = nan_counts[nan_counts > 0]
if len(nan_features) > 0:
    print(f"   Found {len(nan_features)} features with NaN values")
    print(f"   Filling with median values...")
    X = X.fillna(X.median())
else:
    print(f"   No missing values found")

print(f"\n📊 Features for training: {len(feature_cols)}")
print(f"🎯 Target distribution:")
print(f"   Podium: {y.sum()} ({y.mean()*100:.1f}%)")
print(f"   Non-podium: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)\n")

# ============================================
# 2. TRAIN/TEST SPLIT (Stratified by Year)
# ============================================
print("✂️  Splitting data...")

# Use 2021-2024 for training, 2025 for final validation
train_idx = df['year'] < 2025
test_idx = df['year'] == 2025

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print(f"   Training: {len(X_train)} samples (2021-2024)")
print(f"   Testing: {len(X_test)} samples (2025)")
print(f"   Feature count: {X_train.shape[1]}\n")

# Scale features for Neural Network and Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 3. TRAIN MODEL 1: LOGISTIC REGRESSION
# ============================================
print("=" * 80)
print("🔵 MODEL 1: LOGISTIC REGRESSION (Baseline)")
print("=" * 80)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

print(f"✅ Trained!")
print(f"   Accuracy:  {lr_accuracy:.3f}")
print(f"   Precision: {lr_precision:.3f}")
print(f"   Recall:    {lr_recall:.3f}")
print(f"   F1-Score:  {lr_f1:.3f}\n")

# ============================================
# 4. TRAIN MODEL 2: RANDOM FOREST
# ============================================
print("=" * 80)
print("🌲 MODEL 2: RANDOM FOREST")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"✅ Trained!")
print(f"   Accuracy:  {rf_accuracy:.3f}")
print(f"   Precision: {rf_precision:.3f}")
print(f"   Recall:    {rf_recall:.3f}")
print(f"   F1-Score:  {rf_f1:.3f}\n")

# Feature importance
print("🔍 Top 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:40s} {row['importance']:.4f}")
print()

# ============================================
# 5. TRAIN MODEL 3: XGBOOST (with tuning)
# ============================================
print("=" * 80)
print("⚡ MODEL 3: XGBOOST (with GridSearch)")
print("=" * 80)

# Quick GridSearch for hyperparameter tuning
param_grid = {
    'max_depth': [5, 7, 10],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

xgb_base = xgb.XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

print("🔍 Tuning hyperparameters (this takes 2-3 minutes)...")
grid_search = GridSearchCV(
    xgb_base,
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)
xgb_model = grid_search.best_estimator_

print(f"✅ Best parameters: {grid_search.best_params_}")

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print(f"\n✅ Trained!")
print(f"   Accuracy:  {xgb_accuracy:.3f}")
print(f"   Precision: {xgb_precision:.3f}")
print(f"   Recall:    {xgb_recall:.3f}")
print(f"   F1-Score:  {xgb_f1:.3f}\n")

# ============================================
# 6. TRAIN MODEL 4: NEURAL NETWORK
# ============================================
print("=" * 80)
print("🧠 MODEL 4: NEURAL NETWORK (Multi-Layer Perceptron)")
print("=" * 80)

# Build neural network
nn_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("🏗️  Architecture:")
print(f"   Input: {X_train_scaled.shape[1]} features")
print(f"   Hidden Layer 1: 128 neurons (ReLU)")
print(f"   Dropout: 20%")
print(f"   Hidden Layer 2: 64 neurons (ReLU)")
print(f"   Dropout: 20%")
print(f"   Hidden Layer 3: 32 neurons (ReLU)")
print(f"   Output: 1 neuron (Sigmoid)\n")

print("🏋️  Training neural network...")

# Early stopping callback
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=0
)

nn_prob = nn_model.predict(X_test_scaled, verbose=0).flatten()
nn_pred = (nn_prob > 0.5).astype(int)

nn_accuracy = accuracy_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)

print(f"✅ Trained! (stopped at epoch {len(history.history['loss'])})")
print(f"   Accuracy:  {nn_accuracy:.3f}")
print(f"   Precision: {nn_precision:.3f}")
print(f"   Recall:    {nn_recall:.3f}")
print(f"   F1-Score:  {nn_f1:.3f}\n")

# ============================================
# 7. MODEL COMPARISON
# ============================================
print("=" * 80)
print("📊 MODEL COMPARISON - 2025 TEST SET")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Network'],
    'Accuracy': [lr_accuracy, rf_accuracy, xgb_accuracy, nn_accuracy],
    'Precision': [lr_precision, rf_precision, xgb_precision, nn_precision],
    'Recall': [lr_recall, rf_recall, xgb_recall, nn_recall],
    'F1-Score': [lr_f1, rf_f1, xgb_f1, nn_f1]
})

print(results.to_string(index=False))
print()

# Find best model
best_idx = results['Accuracy'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
best_accuracy = results.loc[best_idx, 'Accuracy']

print(f"🏆 BEST MODEL: {best_model_name} ({best_accuracy:.3f} accuracy)")
print()

# ============================================
# 8. SAVE MODELS
# ============================================
print("💾 Saving models...")

import joblib

joblib.dump(lr_model, 'model_logistic_regression.pkl')
joblib.dump(rf_model, 'model_random_forest.pkl')
joblib.dump(xgb_model, 'model_xgboost.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
nn_model.save('model_neural_network.h5')

print("✅ Models saved:")
print("   • model_logistic_regression.pkl")
print("   • model_random_forest.pkl")
print("   • model_xgboost.pkl")
print("   • model_neural_network.h5")
print("   • feature_scaler.pkl")

# ============================================
# 9. VISUALIZATIONS
# ============================================
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Comparison Bar Chart
ax = axes[0, 0]
x = np.arange(len(results))
width = 0.2
ax.bar(x - 1.5*width, results['Accuracy'], width, label='Accuracy', alpha=0.8)
ax.bar(x - 0.5*width, results['Precision'], width, label='Precision', alpha=0.8)
ax.bar(x + 0.5*width, results['Recall'], width, label='Recall', alpha=0.8)
ax.bar(x + 1.5*width, results['F1-Score'], width, label='F1-Score', alpha=0.8)
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(['LogReg', 'RF', 'XGB', 'NN'], rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1])

# 2. Feature Importance (Random Forest)
ax = axes[0, 1]
top_features = feature_importance.head(15)
ax.barh(top_features['feature'], top_features['importance'])
ax.set_xlabel('Importance')
ax.set_title('Top 15 Most Important Features (Random Forest)')
ax.invert_yaxis()

# 3. ROC Curves
ax = axes[1, 0]
models_data = [
    ('Logistic Regression', lr_prob),
    ('Random Forest', rf_prob),
    ('XGBoost', xgb_prob),
    ('Neural Network', nn_prob)
]
for name, prob in models_data:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves')
ax.legend()
ax.grid(alpha=0.3)

# 4. Confusion Matrix (Best Model)
ax = axes[1, 1]
if best_model_name == 'XGBoost':
    cm = confusion_matrix(y_test, xgb_pred)
elif best_model_name == 'Random Forest':
    cm = confusion_matrix(y_test, rf_pred)
elif best_model_name == 'Neural Network':
    cm = confusion_matrix(y_test, nn_pred)
else:
    cm = confusion_matrix(y_test, lr_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {best_model_name}')
ax.set_xticklabels(['No Podium', 'Podium'])
ax.set_yticklabels(['No Podium', 'Podium'])

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison.png")

print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🎯 READY TO PREDICT: 2025 US GRAND PRIX PODIUM")
print(f"   Best model: {best_model_name}")
print(f"   Expected accuracy: {best_accuracy:.1%}")
print("\n🚀 Next: Create prediction script for US GP!")
print("=" * 80)