import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Function to load and preprocess data
def load_data(file) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], list]:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['chance_burnout'])
    y = df['chance_burnout']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y), df.columns.tolist()

# Function to train and evaluate a model
def train_evaluate_model(model, param_grid: Dict[str, Any], X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)

    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_stratified = cross_val_score(best_model, X_scaled, y, cv=stratified_kfold, scoring='accuracy')

    return {
        'accuracy': accuracy,
        'cv_scores': cv_scores.mean(),
        'best_params': grid_search.best_params_,
        'accuracy_best': accuracy_best,
        'cv_scores_stratified': cv_scores_stratified.mean(),
        'best_model': best_model
    }

# Function to train and evaluate Random Forest model
def train_evaluate_rf(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    model_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    param_grid_rf = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    return train_evaluate_model(model_rf, param_grid_rf, X_train, X_test, y_train, y_test, X_scaled, y)

# Function to train and evaluate LightGBM model
def train_evaluate_lgb(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    model_lgb = lgb.LGBMClassifier(n_estimators=500, random_state=42)
    model_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=50
    )
    param_grid_lgb = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [10, 20, 30, -1],
        'num_leaves': [31, 50, 70, 100]
    }
    return train_evaluate_model(model_lgb, param_grid_lgb, X_train, X_test, y_train, y_test, X_scaled, y)

# Function to display results
def display_results(title: str, results: Dict[str, Any]):
    st.subheader(title)
    st.write(f'Accuracy: {results["accuracy"] * 100:.2f}%')
    st.write(f'Cross-validated Accuracy: {results["cv_scores"] * 100:.2f}%')
    st.write(f'Best Parameters: {results["best_params"]}')
    st.write(f'Tuned Accuracy: {results["accuracy_best"] * 100:.2f}%')
    st.write(f'Stratified KFold Accuracy: {results["cv_scores_stratified"] * 100:.2f}%')

# Streamlit app
st.title('Burnout Early Warning System')

st.markdown("""
    **Problem:** College students often experience burnout due to stress, deadlines, and poor work-life balance.

    **Solution:** This ML model predicts burnout risk based on:
    - Sleep patterns (tracked via phone data or self-reports)
    - Assignment deadlines and workload intensity
    - Mood fluctuations (through sentiment analysis in messages/social media interactions)
    - Physiological data (if available, like step count or heart rate)
    - Study habits (time spent on learning resources)

    **Output:** The model gives early warnings and personalized recommendations like relaxation techniques, time management tips, or peer support suggestions.
""")

uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")

if uploaded_file is not None:
    data_load_state = st.text('Loading data...')
    split_data, columns = load_data(uploaded_file)
    data_load_state.text('Loading data... done!')

    st.subheader('Data Columns')
    st.write(columns)

    X_train, X_test, y_train, y_test = split_data
    X_scaled = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    if st.button('Train and Evaluate Models'):
        with st.spinner('Training models...'):
            rf_results = train_evaluate_rf(X_train, X_test, y_train, y_test, X_scaled, y)
            lgb_results = train_evaluate_lgb(X_train, X_test, y_train, y_test, X_scaled, y)

        display_results('Random Forest Results', rf_results)
        display_results('LightGBM Results', lgb_results)

        st.subheader('Feature Importance (LightGBM)')
        lgb.plot_importance(lgb_results['best_model'], max_num_features=10, importance_type='gain', figsize=(10, 6))
        st.pyplot(plt)

        st.subheader('Personalized Recommendations')
        st.markdown("""
            Based on the model's predictions, here are some personalized recommendations:
            - **Relaxation Techniques:** Practice mindfulness, meditation, or deep breathing exercises.
            - **Time Management Tips:** Use tools like Pomodoro Technique to manage your study time effectively.
            - **Peer Support:** Connect with friends or join study groups to share experiences and support each other.
        """)
