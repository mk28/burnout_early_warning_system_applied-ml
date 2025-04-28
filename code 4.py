import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
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

# --- Hyperparameter tuning for RandomForest (expanded search) ---
param_grid_rf = {
    'n_estimators': [100, 200, 300, 500, 1000],  # Increase number of estimators
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': ['sqrt', 'log2', 'auto'],
    'class_weight': ['balanced', None]  # Additional weight options
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_

# --- Hyperparameter tuning for LightGBM (expanded search) ---
param_grid_lgb = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [10, 20, 30, -1],
    'num_leaves': [31, 50, 70, 100, 200],
    'subsample': [0.6, 0.8, 1.0],  # For better generalization
    'colsample_bytree': [0.6, 0.8, 1.0],
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'reg_lambda': [0, 0.1, 1]
}
grid_lgb = GridSearchCV(lgb.LGBMClassifier(random_state=42), param_grid_lgb, cv=5, scoring='accuracy', n_jobs=-1)
grid_lgb.fit(X_train, y_train)
best_lgb = grid_lgb.best_estimator_

# --- Stacking Classifier (Combining RandomForest and LightGBM) ---
stacking_model = StackingClassifier(
    estimators=[('rf', best_rf), ('lgb', best_lgb)],
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42)
)
stacking_model.fit(X_train, y_train)

# --- Evaluate Stacking Classifier ---
y_pred_stacked = stacking_model.predict(X_test)
accuracy_stacked = accuracy_score(y_test, y_pred_stacked)
roc_auc_stacked = roc_auc_score(y_test, stacking_model.predict_proba(X_test), multi_class='ovr')
print(f'Stacking Model Accuracy: {accuracy_stacked * 100:.2f}%')
print(f'Stacking Model ROC AUC: {roc_auc_stacked:.2f}')

# --- Cross-validation on Stacked Model ---
cv_scores_stacked = cross_val_score(stacking_model, X_scaled, y, cv=5, scoring='accuracy')
print(f'Stacked Model Cross-validated Accuracy: {cv_scores_stacked.mean() * 100:.2f}%')

# --- Evaluate Best Random Forest Model ---
y_pred_rf = best_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, best_rf.predict_proba(X_test), multi_class='ovr')
print(f'Best Random Forest Model Accuracy: {accuracy_rf * 100:.2f}%')
print(f'Best Random Forest ROC AUC: {roc_auc_rf:.2f}')

# --- Evaluate Best LightGBM Model ---
y_pred_lgb = best_lgb.predict(X_test)
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
roc_auc_lgb = roc_auc_score(y_test, best_lgb.predict_proba(X_test), multi_class='ovr')
print(f'Best LightGBM Model Accuracy: {accuracy_lgb * 100:.2f}%')
print(f'Best LightGBM ROC AUC: {roc_auc_lgb:.2f}')

# --- Plot feature importance for LightGBM ---
lgb.plot_importance(best_lgb, max_num_features=10, importance_type='gain', figsize=(10, 6))
plt.title('Top 10 Feature Importances (LightGBM)')
plt.show()





