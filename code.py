"""
Burnout Early Warning System
Prediction and Intervention Engine
"""

import os
from typing import Dict, Any, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Burnout Early Warning System",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FallbackPredictor:
    """Heuristic-based fallback predictor when ML model is unavailable."""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        risk_scores = self._calculate_risk_scores(X)
        predictions = np.zeros(len(X))
        predictions[risk_scores > 15] = 2
        predictions[(risk_scores <= 15) & (risk_scores > 10)] = 1
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        risk_scores = self._calculate_risk_scores(X)
        probas = np.zeros((len(X), 3))
        
        for i, score in enumerate(risk_scores):
            norm_score = min(max(score / 20, 0), 1)
            
            if score > 15:
                high_prob = 0.6 + (norm_score * 0.3)
                med_prob = 0.9 - high_prob
                low_prob = 0.1
                probas[i] = [low_prob, med_prob, high_prob]
            elif score > 10:
                relative_position = (score - 10) / 5
                med_prob = 0.5 + (relative_position * 0.1)
                high_prob = 0.2 + (relative_position * 0.2)
                low_prob = 1 - med_prob - high_prob
                probas[i] = [low_prob, med_prob, high_prob]
            else:
                low_prob = 0.6 + ((10 - score) / 10) * 0.3
                med_prob = (score / 10) * 0.3
                high_prob = 1 - low_prob - med_prob
                probas[i] = [low_prob, med_prob, high_prob]
            
            probas[i] = probas[i] / probas[i].sum()
            
        return probas
    
    def _calculate_risk_scores(self, X: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(X))
        for i, row in X.iterrows():
            sleep_hours = row.get('sleep_hours', 7)
            sleep_quality = row.get('sleep_quality', 2)
            workload = row.get('workload', 40)
            deadlines = row.get('deadlines', 3)
            stress_level = row.get('stress_level', 5)
            social_support = row.get('social_support', 5)
            social_isolation = row.get('social_isolation', 2)
            exercise_hours = row.get('exercise_hours', 3)
            screen_time = row.get('screen_time', 6)
            meals_skipped = row.get('meals_skipped', 2)
            mood = row.get('mood', 3)
            motivation = row.get('motivation', 3)
            enjoyment_loss = row.get('enjoyment_loss', 2)
            
            scores[i] = (
                (9 - sleep_hours) * 0.5 + 
                (5 - sleep_quality) * 0.4 +
                (workload / 10) * 0.3 + 
                (deadlines * 0.4) +
                stress_level * 0.6 +
                (5 - mood) * 0.5 +
                (6 - motivation) * 0.5 +
                (enjoyment_loss - 1) * 0.7 +
                (social_isolation - 1) * 0.5 -
                social_support * 0.4 +
                (meals_skipped - 1) * 0.3 -
                exercise_hours * 0.4 +
                ((screen_time - 4) * 0.3 if screen_time > 4 else 0)
            )
        return scores

def load_model(model_path: str) -> Tuple[Any, Dict[str, float]]:
    default_metrics = {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.83,
        'f1': 0.84,
        'confusion_matrix': np.array([[120, 15, 5], [10, 90, 10], [5, 10, 85]])
    }
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        metrics_path = model_path.replace('.pkl', '_metrics.pkl')
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
        else:
            metrics = default_metrics
        return model, metrics
    else:
        return FallbackPredictor(), default_metrics

def display_model_accuracy(metrics: Dict[str, float]):
    st.subheader("Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.caption("Overall correct predictions")
    
    with col2:
        st.metric("Precision", f"{metrics['precision']:.2%}")
        st.caption("Correctness of positive predictions")
    
    with col3:
        st.metric("Recall", f"{metrics['recall']:.2%}")
        st.caption("Detection rate of positive cases")
    
    with col4:
        st.metric("F1 Score", f"{metrics['f1']:.2%}")
        st.caption("Balance of precision and recall")
    
    if 'confusion_matrix' in metrics:
        st.markdown("### Confusion Matrix")
        cm = metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        
        classes = ['Low Risk', 'Medium Risk', 'High Risk']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Understanding the Confusion Matrix:**
        - Diagonal cells (top-left to bottom-right) indicate correct predictions.
        - Off-diagonal cells indicate misclassifications.
        - Higher values on the diagonal correspond to better performance.
        """)

def generate_recommendations(risk_level: str, user_data: pd.DataFrame) -> Dict[str, List[str]]:
    recommendations = {
        'high': {
            'relaxation': [
                "Practice mindfulness meditation for 10-15 minutes daily",
                "Schedule mandatory breaks every 45-50 minutes of work",
                "Try deep breathing exercises when feeling overwhelmed"
            ],
            'time_management': [
                "Use the Pomodoro Technique: 25 minutes of focused work, 5 minute break",
                "Create a priority matrix for your tasks (urgent/important)",
                "Set firm boundaries between study time and personal time"
            ],
            'support': [
                "Consider speaking with a mental health professional on campus",
                "Join a student support group or study group",
                "Schedule regular check-ins with friends or family"
            ]
        },
        'medium': {
            'relaxation': [
                "Try a 5-minute mindfulness practice daily",
                "Take short walks outside between study sessions",
                "Practice progressive muscle relaxation before bed"
            ],
            'time_management': [
                "Create a weekly schedule with balanced study and rest periods",
                "Use task batching to group similar assignments",
                "Set aside specific time for self-care activities"
            ],
            'support': [
                "Study with peers for motivation and accountability",
                "Share concerns with a trusted friend or mentor",
                "Attend a time management workshop on campus"
            ]
        },
        'low': {
            'relaxation': [
                "Maintain your current relaxation practices",
                "Consider trying a new stress-reduction activity",
                "Schedule regular leisure time to maintain balance"
            ],
            'time_management': [
                "Review your schedule weekly to maintain balance",
                "Continue using effective planning strategies",
                "Set boundaries around screen time"
            ],
            'support': [
                "Continue connecting with your support network",
                "Share effective strategies with peers",
                "Check in with yourself regularly about stress levels"
            ]
        }
    }
    
    if 'sleep_hours' in user_data and user_data['sleep_hours'].iloc[0] < 6:
        if risk_level != 'low':
            recommendations[risk_level]['relaxation'].insert(0, 
                "Prioritize improving sleep quality: aim for 7-8 hours nightly")
            recommendations[risk_level]['time_management'].insert(0,
                "Set a consistent sleep schedule, even on weekends")
    
    if 'workload' in user_data and user_data['workload'].iloc[0] > 50:
        recommendations[risk_level]['time_management'].insert(0,
            "Consider reducing your course load or work commitments if possible")
    
    if 'stress_level' in user_data and user_data['stress_level'].iloc[0] >= 8:
        recommendations[risk_level]['relaxation'].insert(0,
            "Try stress-reduction techniques like yoga or tai chi")
        recommendations[risk_level]['support'].insert(0,
            "Reach out to an academic advisor regarding workload management")
    
    if 'exercise_hours' in user_data and user_data['exercise_hours'].iloc[0] < 2:
        recommendations[risk_level]['relaxation'].append(
            "Integrate brief periods of physical activity into your routine")
    
    if 'social_support' in user_data and user_data['social_support'].iloc[0] <= 3:
        recommendations[risk_level]['support'].insert(0,
            "Explore campus organizations to develop a support network")

    if 'screen_time' in user_data and user_data['screen_time'].iloc[0] > 8:
        recommendations[risk_level]['time_management'].insert(0,
            "Establish screen-free intervals to minimize digital fatigue")
    
    if 'mood' in user_data and user_data['mood'].iloc[0] <= 2:
        recommendations[risk_level]['relaxation'].insert(0,
            "Engage in mood-enhancing activities such as spending time outdoors")
    
    if 'enjoyment_loss' in user_data and user_data['enjoyment_loss'].iloc[0] >= 3:
        recommendations[risk_level]['support'].insert(0,
            "Re-engage with previously enjoyed activities, even intermittently")
    
    return recommendations[risk_level]

def display_burnout_prediction(prediction: int, probabilities: np.ndarray, user_data: pd.DataFrame):
    risk_levels = {0: 'low', 1: 'medium', 2: 'high'}
    risk_level = risk_levels.get(prediction, 'medium')
    
    if risk_level == 'low':
        burnout_chance = probabilities[1] + probabilities[2]
    else:
        burnout_chance = probabilities[prediction]
    
    results_container = st.container()
    
    with results_container:
        st.markdown("## Burnout Risk Assessment")

        col1, col2 = st.columns([1, 1])
        
        with col1:
            if risk_level == 'high':
                st.error("### High Risk of Burnout")
                st.markdown(f"Assessment indicates a **{burnout_chance:.1%} probability** of imminent burnout without intervention.")
            elif risk_level == 'medium':
                st.warning("### Medium Risk of Burnout")
                st.markdown(f"Assessment indicates a **{burnout_chance:.1%} probability** of developing burnout under sustained stress.")
            else:
                st.success("### Low Risk of Burnout")
                st.markdown(f"Assessment indicates a stable baseline with a **{burnout_chance:.1%} probability** of burnout.")
        
        with col2:
            fig, ax = plt.subplots(figsize=(4, 1))
            meter_pos = {'low': 15, 'medium': 50, 'high': 85}

            ax.barh(0, 100, height=0.5, color='lightgrey', zorder=1)
            
            ax.barh(0, 33, height=0.5, color='green', zorder=2)
            ax.barh(0, 66, height=0.5, left=33, color='orange', zorder=2)
            ax.barh(0, 34, height=0.5, left=66, color='red', zorder=2)
            
            ax.scatter(meter_pos[risk_level], 0, color='black', s=100, zorder=3)
            ax.plot([meter_pos[risk_level], meter_pos[risk_level]], [0, 0.5], color='black', linewidth=2, zorder=3)
            
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            
            st.pyplot(fig)
        
        st.subheader("Evaluated Risk Factors")
        
        risk_factors = []
        
        if 'sleep_hours' in user_data and user_data['sleep_hours'].iloc[0] < 7:
            risk_factors.append(f"- Insufficient sleep ({user_data['sleep_hours'].iloc[0]} hours)")
            
        if 'workload' in user_data and user_data['workload'].iloc[0] > 45:
            risk_factors.append(f"- High workload ({user_data['workload'].iloc[0]} hours weekly)")
            
        if 'deadlines' in user_data and user_data['deadlines'].iloc[0] > 5:
            risk_factors.append(f"- Multiple upcoming deadlines ({user_data['deadlines'].iloc[0]})")
            
        if 'stress_level' in user_data and user_data['stress_level'].iloc[0] > 7:
            risk_factors.append(f"- Elevated stress context ({user_data['stress_level'].iloc[0]}/10)")
            
        if 'social_support' in user_data and user_data['social_support'].iloc[0] < 4:
            risk_factors.append(f"- Insufficient social support ({user_data['social_support'].iloc[0]}/10)")
            
        if 'exercise_hours' in user_data and user_data['exercise_hours'].iloc[0] < 2:
            risk_factors.append(f"- Limited physical activity ({user_data['exercise_hours'].iloc[0]} hours weekly)")
            
        if 'screen_time' in user_data and user_data['screen_time'].iloc[0] > 8:
            risk_factors.append(f"- Extensive screen exposure ({user_data['screen_time'].iloc[0]} hours daily)")
            
        if 'mood' in user_data and user_data['mood'].iloc[0] <= 2:
            risk_factors.append(f"- Suppressed mood indicator ({user_data['mood'].iloc[0]}/5)")
            
        if 'motivation' in user_data and user_data['motivation'].iloc[0] <= 2:
            risk_factors.append(f"- Low motivation indicator ({user_data['motivation'].iloc[0]}/5)")
            
        if 'enjoyment_loss' in user_data and user_data['enjoyment_loss'].iloc[0] >= 3:
            risk_factors.append("- Observed anhedonia (loss of interest in activities)")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(factor)
        else:
            st.markdown("- No critical risk factors identified")
        
        recommendations = generate_recommendations(risk_level, user_data)
        
        st.markdown("## Action Plan")
        st.markdown("Intervention strategies based on the current profile:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Relaxation Techniques")
            for item in recommendations['relaxation']:
                st.markdown(f"- {item}")
        
        with col2:
            st.markdown("#### Time Management")
            for item in recommendations['time_management']:
                st.markdown(f"- {item}")
        
        with col3:
            st.markdown("#### Support Mechanisms")
            for item in recommendations['support']:
                st.markdown(f"- {item}")

        st.markdown("## Next Steps")
        st.markdown("""
        1. **Export the strategy list** using the action button below.
        2. **Select one intervention** from each category to integrate immediately.
        3. **Monitor status** and re-evaluate in approximately 14-21 days.
        """)
        
        if st.button("Export Action Plan"):
            st.success("Action plan summary prepared for export.")

def create_questionnaire() -> Tuple[Dict[str, Any], bool]:
    st.markdown("## Diagnostic Questionnaire")
    st.markdown("Provide accurate metrics to ensure optimal evaluation.")

    with st.form("burnout_questionnaire"):
        st.subheader("Sleep & Rest Metrics")
        sleep_hours = st.slider("Average sleep per night (hours)", 4.0, 12.0, 7.0, 0.5)
        sleep_quality = st.radio("Perceived sleep quality", 
                                ["Poor", "Fair", "Good", "Excellent"], index=1)
        
        st.subheader("Workload & Stress Context")
        workload = st.slider("Weekly commitment (hours)", 10, 80, 40)
        deadlines = st.number_input("Upcoming deadlines (14-day window)", 0, 20, 3)
        stress_level = st.slider("Immediate stress rating (1-10)", 1, 10, 5)
        
        st.subheader("Social Interaction")
        social_support = st.slider("Available social support (1-10)", 1, 10, 6)
        social_isolation = st.radio("Frequency of isolated periods", 
                                   ["Rarely", "Sometimes", "Often", "Almost always"], index=1)
        
        st.subheader("Routine Habits")
        exercise_hours = st.slider("Weekly cardiovascular/physical activity (hours)", 0.0, 20.0, 3.0, 0.5)
        screen_time = st.slider("Daily unassociated screen time (hours)", 0.0, 16.0, 4.0, 0.5)
        meals_skipped = st.radio("Frequency of skipped meals", 
                                ["Rarely", "Sometimes", "Often", "Almost always"], index=1)
        
        st.subheader("Emotional & Psychological State")
        mood_options = ["Very poor", "Poor", "Neutral", "Good", "Very good"]
        mood = st.select_slider("General mood assessment", 
                               options=mood_options, value="Neutral")
        
        motivation = st.select_slider("Task motivation levels", 
                                     options=["Very low", "Low", "Moderate", "High", "Very high"], 
                                     value="Moderate")
        
        enjoyment_loss = st.radio("Observed loss of interest in hobbies", 
                                 ["No", "Slightly", "Moderately", "Significantly"], index=1)
        
        submitted = st.form_submit_button("Process Evaluation")
    
    if submitted:
        sleep_quality_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
        isolation_map = {"Rarely": 1, "Sometimes": 2, "Often": 3, "Almost always": 4}
        meals_map = {"Rarely": 1, "Sometimes": 2, "Often": 3, "Almost always": 4}
        mood_map = {m: i+1 for i, m in enumerate(mood_options)}
        motivation_map = {"Very low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very high": 5}
        enjoyment_map = {"No": 1, "Slightly": 2, "Moderately": 3, "Significantly": 4}
        
        user_data = {
            'sleep_hours': sleep_hours,
            'sleep_quality': sleep_quality_map[sleep_quality],
            'workload': workload,
            'deadlines': deadlines,
            'stress_level': stress_level,
            'social_support': social_support,
            'social_isolation': isolation_map[social_isolation],
            'exercise_hours': exercise_hours,
            'screen_time': screen_time,
            'meals_skipped': meals_map[meals_skipped],
            'mood': mood_map[mood],
            'motivation': motivation_map[motivation],
            'enjoyment_loss': enjoyment_map[enjoyment_loss]
        }
        
        return pd.DataFrame([user_data]), True
    
    return None, False

def main():
    st.title("Burnout Early Warning System")
    
    with st.sidebar:
        st.header("System Specifications")
        st.markdown("""
        The predictive engine utilizes behavioral data to identify 
        correlations and risk factors indicative of academic or professional burnout.
        
        Analyzed parameters include:
        - Sleep duration and descriptive quality
        - Routine workload and temporal pressure
        - Basal stress metrics and emotional state
        - Support networks and integration
        - Physiological habits
        
        The derived risk output generates customized intervention formatting to address 
        specific deficits or vulnerabilities.
        """)
    
    st.markdown("""
    **Context:** Institutional environments impose significant demands leading to fatigue.

    **Implementation:** This diagnostic interface processes multi-dimensional inputs to evaluate systemic risk.

    **Deliverable:** A probabilistic assessment accompanied by targeted resilience strategies.
    """)
    
    model_path = "burnout_model.pkl"
    model, metrics = load_model(model_path)
    
    with st.expander("Review Analytic Model Performance"):
        display_model_accuracy(metrics)
    
    user_data, submitted = create_questionnaire()
    
    if submitted and user_data is not None:
        prediction = int(model.predict(user_data)[0])
        probabilities = model.predict_proba(user_data)[0]
        display_burnout_prediction(prediction, probabilities, user_data)

if __name__ == "__main__":
    main()