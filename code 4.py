import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt

# Step 1: Load your dataset
df = pd.read_csv(r'C:\Users\kabdw\OneDrive\Desktop\AIML CODE\Project ML - Sheet1.csv')

# Step 2: Fix column names (remove spaces)
df.columns = df.columns.str.strip()

print("Columns after stripping spaces:", df.columns.tolist())

# Step 3: Drop columns with all NaN values
df = df.dropna(axis=1, how='all')

# Step 4: Fill missing values
# 4.1 Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 4.2 Impute numeric columns with median
numeric_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

# 4.3 Impute categorical columns with most frequent
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Step 5: Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 6: Prepare features and target
X = df.drop(columns=['chance_burnout'])
y = df['chance_burnout']

# Step 7: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- Random Forest Model ---
# Step 9: Train RandomForestClassifier
model_rf = RandomForestClassifier(
    n_estimators=500,  # Increase number of estimators to improve accuracy
    max_depth=None,  # Let the tree grow fully to capture more patterns
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # Test with different features per split
    class_weight='balanced',  # Handle imbalance if needed
    random_state=42
)
model_rf.fit(X_train, y_train)

# Step 10: Evaluate Random Forest
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%')

# Step 11: Cross-validation for Random Forest
cv_scores_rf = cross_val_score(model_rf, X_scaled, y, cv=5, scoring='accuracy')
print(f'Random Forest Cross-validated Accuracy: {cv_scores_rf.mean() * 100:.2f}%')

# Step 12: Hyperparameter tuning Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500, 1000],  # Increase number of estimators
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]  # Testing different max features
}
grid_rf = GridSearchCV(model_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print(f'Best parameters for Random Forest: {grid_rf.best_params_}')
best_rf = grid_rf.best_estimator_

# Step 13: Evaluate tuned Random Forest
y_pred_best_rf = best_rf.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
print(f'Tuned Random Forest Accuracy: {accuracy_best_rf * 100:.2f}%')

# Step 14: Stratified KFold Cross-validation Random Forest
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_stratified_rf = cross_val_score(best_rf, X_scaled, y, cv=stratified_kfold, scoring='accuracy')
print(f'Stratified KFold Accuracy (Random Forest): {cv_scores_stratified_rf.mean() * 100:.2f}%')

# --- LightGBM Model ---
# Step 15: Train LightGBM
model_lgb = lgb.LGBMClassifier(n_estimators=500, random_state=42)
model_lgb.fit(
    X_train, 
    y_train, 
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,  # Early stopping to avoid overfitting
    verbose=50
)

# Step 16: Evaluate LightGBM
y_pred_lgb = model_lgb.predict(X_test)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print(f'LightGBM Model Accuracy: {accuracy_lgb * 100:.2f}%')

# Step 17: Cross-validation LightGBM
cv_scores_lgb = cross_val_score(model_lgb, X_scaled, y, cv=5, scoring='accuracy')
print(f'LightGBM Cross-validated Accuracy: {cv_scores_lgb.mean() * 100:.2f}%')

# Step 18: Hyperparameter tuning LightGBM
param_grid_lgb = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [10, 20, 30, -1],  # Try with no max depth
    'num_leaves': [31, 50, 70, 100]  # More leaves can sometimes improve accuracy
}
grid_lgb = GridSearchCV(model_lgb, param_grid_lgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_lgb.fit(X_train, y_train)

print(f'Best parameters for LightGBM: {grid_lgb.best_params_}')
best_lgb = grid_lgb.best_estimator_

# Step 19: Evaluate tuned LightGBM
y_pred_best_lgb = best_lgb.predict(X_test)
accuracy_best_lgb = accuracy_score(y_test, y_pred_best_lgb)
print(f'Tuned LightGBM Accuracy: {accuracy_best_lgb * 100:.2f}%')

# Step 20: Stratified KFold Cross-validation LightGBM
cv_scores_stratified_lgb = cross_val_score(best_lgb, X_scaled, y, cv=stratified_kfold, scoring='accuracy')
print(f'Stratified KFold Accuracy (LightGBM): {cv_scores_stratified_lgb.mean() * 100:.2f}%')

# Step 21: Plot feature importance (LightGBM)
lgb.plot_importance(best_lgb, max_num_features=10, importance_type='gain', figsize=(10, 6))
plt.title('Top 10 Feature Importances (LightGBM)')
plt.show()


