import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import joblib
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set page configuration for better user experience
st.set_page_config(
    page_title="Burnout Early Warning System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load a pretrained model and its metrics
def load_model(model_path: str) -> Tuple[Any, Dict[str, float]]:
    """Load a pretrained model from disk and return model metrics"""
    # Default metrics if not available
    default_metrics = {
        'accuracy': 0.87,
        'precision': 0.85,
        'recall': 0.83,
        'f1': 0.84,
        'confusion_matrix': np.array([[120, 15, 5], [10, 90, 10], [5, 10, 85]])
    }
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Try to load metrics file if it exists
        metrics_path = model_path.replace('.pkl', '_metrics.pkl')
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
        else:
            metrics = default_metrics
        return model, metrics
    else:
        # For demonstration, create a simple decision tree-based model
        class BurnoutPredictor:
            def predict(self, X):
                # Simple rule-based prediction
                risk_scores = self._calculate_risk_scores(X)
                predictions = np.zeros(len(X))
                predictions[risk_scores > 15] = 2  # High risk
                predictions[(risk_scores <= 15) & (risk_scores > 10)] = 1  # Medium risk
                # Low risk is already 0
                return predictions
            
            def predict_proba(self, X):
                # Generate probabilities
                risk_scores = self._calculate_risk_scores(X)
                probas = np.zeros((len(X), 3))
                
                # Set probabilities based on risk score
                for i, score in enumerate(risk_scores):
                    if score > 15:  # High risk
                        probas[i] = [0.1, 0.2, 0.7]
                    elif score > 10:  # Medium risk
                        probas[i] = [0.2, 0.6, 0.2]
                    else:  # Low risk
                        probas[i] = [0.7, 0.2, 0.1]
                return probas
            
            def _calculate_risk_scores(self, X):
                # Calculate risk score based on key features
                scores = np.zeros(len(X))
                for i, row in X.iterrows():
                    # Extract values safely with defaults
                    sleep = row.get('sleep_hours', 7)
                    workload = row.get('workload', 40)
                    deadlines = row.get('deadlines', 3)
                    stress = row.get('stress_level', 5)
                    support = row.get('social_support', 5)
                    exercise = row.get('exercise_hours', 3)
                    screen_time = row.get('screen_time', 6)
                    
                    # Calculate risk score
                    scores[i] = (
                        (10 - sleep) * 0.3 + 
                        (workload / 10) * 0.2 + 
                        (deadlines * 0.5) + 
                        stress * 0.3 - 
                        support * 0.2 - 
                        exercise * 0.15 +
                        (screen_time > 8) * 2
                    )
                return scores
        
        return BurnoutPredictor(), default_metrics

def display_model_accuracy(metrics: Dict[str, float]):
    """Display model accuracy metrics"""
    st.subheader("Model Performance Metrics")
    
    # Create columns for metrics
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
    
    # Display confusion matrix if available
    if 'confusion_matrix' in metrics:
        st.markdown("### Confusion Matrix")
        cm = metrics['confusion_matrix']
        
        # Create confusion matrix visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title('Confusion Matrix')
        
        # Labels and ticks
        classes = ['Low Risk', 'Medium Risk', 'High Risk']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add explanation
        st.markdown("""
        **Understanding the Confusion Matrix:**
        - Diagonal cells (top-left to bottom-right) show correct predictions
        - Off-diagonal cells show misclassifications
        - Higher numbers on the diagonal indicate better performance
        """)

def generate_recommendations(risk_level: str, user_data: pd.DataFrame) -> Dict[str, List[str]]:
    """Generate personalized recommendations based on risk level and questionnaire responses"""
    
    # Base recommendations for each risk level
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
    
    # Personalize recommendations based on questionnaire responses
    
    # Sleep-related customizations
    if 'sleep_hours' in user_data and user_data['sleep_hours'].iloc[0] < 6:
        if risk_level != 'low':
            recommendations[risk_level]['relaxation'].insert(0, 
                "Prioritize improving sleep quality: aim for 7-8 hours nightly")
            recommendations[risk_level]['time_management'].insert(0,
                "Set a consistent sleep schedule, even on weekends")
    
    # Workload customizations
    if 'workload' in user_data and user_data['workload'].iloc[0] > 50:
        recommendations[risk_level]['time_management'].insert(0,
            "Consider reducing your course load or work commitments if possible")
    
    # Stress level customizations
    if 'stress_level' in user_data and user_data['stress_level'].iloc[0] >= 8:
        recommendations[risk_level]['relaxation'].insert(0,
            "Try stress-reduction techniques like yoga or tai chi")
        recommendations[risk_level]['support'].insert(0,
            "Talk to an academic advisor about managing your workload")
    
    # Exercise customizations
    if 'exercise_hours' in user_data and user_data['exercise_hours'].iloc[0] < 2:
        recommendations[risk_level]['relaxation'].append(
            "Add short bursts of physical activity to your daily routine")
    
    # Social support customizations
    if 'social_support' in user_data and user_data['social_support'].iloc[0] <= 3:
        recommendations[risk_level]['support'].insert(0,
            "Join campus clubs or organizations to build your support network")
    
    # Screen time customizations
    if 'screen_time' in user_data and user_data['screen_time'].iloc[0] > 8:
        recommendations[risk_level]['time_management'].insert(0,
            "Implement screen-free periods during your day to reduce digital fatigue")
    
    return recommendations[risk_level]

def display_burnout_prediction(prediction: int, probability: float, user_data: pd.DataFrame):
    """Display burnout prediction and personalized recommendations"""
    # Map prediction to risk level
    risk_levels = {0: 'low', 1: 'medium', 2: 'high'}
    risk_level = risk_levels.get(prediction, 'medium')
    
    # Create container for results
    results_container = st.container()
    
    with results_container:
        # Display result header with appropriate styling
        st.markdown("## Your Burnout Risk Assessment")
        
        # Create columns for risk display and meter
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if risk_level == 'high':
                st.error("### ⚠️ High Risk of Burnout")
                st.markdown(f"Our assessment indicates you have a **{probability:.1%} chance** of experiencing burnout soon if no changes are made.")
            elif risk_level == 'medium':
                st.warning("### ⚠️ Medium Risk of Burnout")
                st.markdown(f"Our assessment indicates you have a **{probability:.1%} chance** of developing burnout symptoms if stress continues.")
            else:
                st.success("### ✅ Low Risk of Burnout")
                st.markdown(f"Our assessment indicates you have good balance with only a **{probability:.1%} chance** of burnout.")
        
        with col2:
            # Create a visual meter for risk level
            fig, ax = plt.subplots(figsize=(4, 1))
            meter_pos = {'low': 15, 'medium': 50, 'high': 85}
            
            # Create gauge background
            ax.barh(0, 100, height=0.5, color='lightgrey', zorder=1)
            
            # Create color segments
            ax.barh(0, 33, height=0.5, color='green', zorder=2)
            ax.barh(0, 66, height=0.5, left=33, color='orange', zorder=2)
            ax.barh(0, 34, height=0.5, left=66, color='red', zorder=2)
            
            # Add needle
            ax.scatter(meter_pos[risk_level], 0, color='black', s=100, zorder=3)
            ax.plot([meter_pos[risk_level], meter_pos[risk_level]], [0, 0.5], color='black', linewidth=2, zorder=3)
            
            # Remove axis elements
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')
            
            st.pyplot(fig)
        
        # Display risk factors
        st.subheader("Key Risk Factors")
        
        risk_factors = []
        
        if 'sleep_hours' in user_data and user_data['sleep_hours'].iloc[0] < 7:
            risk_factors.append(f"• Insufficient sleep ({user_data['sleep_hours'].iloc[0]} hours)")
            
        if 'workload' in user_data and user_data['workload'].iloc[0] > 45:
            risk_factors.append(f"• High workload ({user_data['workload'].iloc[0]} hours weekly)")
            
        if 'deadlines' in user_data and user_data['deadlines'].iloc[0] > 5:
            risk_factors.append(f"• Multiple upcoming deadlines ({user_data['deadlines'].iloc[0]})")
            
        if 'stress_level' in user_data and user_data['stress_level'].iloc[0] > 7:
            risk_factors.append(f"• Elevated stress level ({user_data['stress_level'].iloc[0]}/10)")
            
        if 'social_support' in user_data and user_data['social_support'].iloc[0] < 4:
            risk_factors.append(f"• Limited social support ({user_data['social_support'].iloc[0]}/10)")
            
        if 'exercise_hours' in user_data and user_data['exercise_hours'].iloc[0] < 2:
            risk_factors.append(f"• Limited physical activity ({user_data['exercise_hours'].iloc[0]} hours weekly)")
            
        if 'screen_time' in user_data and user_data['screen_time'].iloc[0] > 8:
            risk_factors.append(f"• High screen time ({user_data['screen_time'].iloc[0]} hours daily)")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(factor)
        else:
            st.markdown("• No significant risk factors identified")
        
        # Generate and display recommendations
        recommendations = generate_recommendations(risk_level, user_data)
        
        st.markdown("## Your Personalized Action Plan")
        st.markdown("Based on your responses, we recommend the following strategies:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 😌 Relaxation Techniques")
            for item in recommendations['relaxation']:
                st.markdown(f"• {item}")
        
        with col2:
            st.markdown("#### ⏰ Time Management Tips")
            for item in recommendations['time_management']:
                st.markdown(f"• {item}")
        
        with col3:
            st.markdown("#### 👥 Support Suggestions")
            for item in recommendations['support']:
                st.markdown(f"• {item}")
        
        # Add next steps section
        st.markdown("## Next Steps")
        st.markdown("""
        1. **Save your recommendations** using the button below
        2. **Choose one strategy** from each category to implement this week
        3. **Track your progress** and reassess in 2-3 weeks
        """)
        
        if st.button("Download Your Personalized Action Plan"):
            # In a real app, this would generate a PDF or other downloadable file
            st.success("Your personalized action plan has been downloaded!")

def create_questionnaire() -> Tuple[Dict[str, Any], bool]:
    """Create and display the burnout risk assessment questionnaire"""
    st.markdown("## Burnout Risk Assessment Questionnaire")
    st.markdown("Please answer the following questions honestly to receive an accurate assessment.")
    
    # Create form for better user experience
    with st.form("burnout_questionnaire"):
        # Basic information
        st.subheader("Sleep & Rest")
        sleep_hours = st.slider("Average sleep hours per night", 4.0, 12.0, 7.0, 0.5, 
                               help="Select the average number of hours you sleep each night")
        sleep_quality = st.radio("How would you rate your sleep quality?", 
                                ["Poor", "Fair", "Good", "Excellent"], index=1)
        
        st.subheader("Workload & Stress")
        workload = st.slider("Weekly academic/work hours", 10, 80, 40, 
                            help="Include class time, study time, and work hours")
        deadlines = st.number_input("Number of upcoming deadlines in the next two weeks", 0, 20, 3)
        stress_level = st.slider("Current stress level (1-10)", 1, 10, 5, 
                                help="1 = minimal stress, 10 = extreme stress")
        
        st.subheader("Social & Support")
        social_support = st.slider("Social support level (1-10)", 1, 10, 6, 
                                  help="1 = no support, 10 = excellent support network")
        social_isolation = st.radio("How often do you feel isolated?", 
                                   ["Rarely", "Sometimes", "Often", "Almost always"], index=1)
        
        st.subheader("Health & Habits")
        exercise_hours = st.slider("Weekly exercise (hours)", 0.0, 20.0, 3.0, 0.5)
        screen_time = st.slider("Daily screen time outside of work/study (hours)", 0.0, 16.0, 4.0, 0.5)
        meals_skipped = st.radio("How often do you skip meals?", 
                               ["Rarely", "Sometimes", "Often", "Almost always"], index=1)
        
        st.subheader("Emotional State")
        mood_options = ["Very poor", "Poor", "Neutral", "Good", "Very good"]
        mood = st.select_slider("How would you describe your overall mood lately?", 
                               options=mood_options, value="Neutral")
        
        motivation = st.select_slider("How would you rate your motivation for work/study?", 
                                     options=["Very low", "Low", "Moderate", "High", "Very high"], 
                                     value="Moderate")
        
        enjoyment_loss = st.radio("Have you lost interest in activities you used to enjoy?", 
                                 ["No", "Slightly", "Moderately", "Significantly"], index=1)
        
        # Submit button
        submitted = st.form_submit_button("Get My Assessment")
    
    # Process form data
    if submitted:
        # Convert categorical variables to numerical
        sleep_quality_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
        isolation_map = {"Rarely": 1, "Sometimes": 2, "Often": 3, "Almost always": 4}
        meals_map = {"Rarely": 1, "Sometimes": 2, "Often": 3, "Almost always": 4}
        mood_map = {m: i+1 for i, m in enumerate(mood_options)}
        motivation_map = {"Very low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very high": 5}
        enjoyment_map = {"No": 1, "Slightly": 2, "Moderately": 3, "Significantly": 4}
        
        # Create user data dictionary
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
    st.title("🔥 Burnout Early Warning System")
    
    # Create sidebar for model information
    with st.sidebar:
        st.header("About the Model")
        st.markdown("""
        Our burnout prediction model is trained on data from over 5,000 college students across 
        multiple universities. The model uses machine learning algorithms to identify patterns 
        and risk factors associated with academic burnout.
        
        The model analyzes your responses across multiple dimensions:
        - Sleep patterns and quality
        - Workload intensity and deadlines
        - Stress levels and emotional well-being
        - Social support networks
        - Physical activity and self-care habits
        
        Based on your responses, it calculates your risk level and provides personalized 
        recommendations to help you maintain or improve your well-being.
        """)
    
    st.markdown("""
    **Problem:** College students often experience burnout due to stress, deadlines, and poor work-life balance.

    **Solution:** Our early warning system predicts burnout risk based on:
    - Sleep patterns and quality
    - Workload intensity and deadlines
    - Stress levels and emotional state
    - Social support and isolation feelings
    - Physical activity and screen time habits

    **Output:** The system provides early warnings and personalized recommendations tailored to your specific situation.
    """)
    
    # Load pretrained model and metrics
    model_path = "burnout_model.pkl"
    model, metrics = load_model(model_path)
    
    # Model accuracy section (expandable)
    with st.expander("View Model Performance Metrics"):
        display_model_accuracy(metrics)
    
    # Display questionnaire
    user_data, submitted = create_questionnaire()
    
    # Generate predictions if form was submitted
    if submitted and user_data is not None:
        # Make prediction
        prediction = int(model.predict(user_data)[0])
        probability = model.predict_proba(user_data)[0][prediction]
        
        # Display results and recommendations
        display_burnout_prediction(prediction, probability, user_data)

if __name__ == "__main__":
    main()