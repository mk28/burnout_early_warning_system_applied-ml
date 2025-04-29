""" Streamlit frontend for Burnout Early Warning System """

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import backend  # Import the backend functions

st.set_page_config(
    page_title="Burnout Early Warning System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def display_model_accuracy(metrics: Dict[str, float]):
    """Display model performance metrics."""

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
        - Diagonal cells (top-left to bottom-right) show correct predictions
        - Off-diagonal cells show misclassifications
        - Higher numbers on the diagonal indicate better performance
        """)


def display_burnout_prediction(prediction: int, probability: float, user_data: pd.DataFrame):
    """Display burnout prediction and personalized recommendations"""

    risk_levels = {0: 'low', 1: 'medium', 2: 'high'}
    risk_level = risk_levels.get(prediction, 'medium')

    results_container = st.container()

    with results_container:
        st.markdown("## Your Burnout Risk Assessment")

        col1, col2 = st.columns([1, 1])

        with col1:
            if risk_level == 'high':
                st.error("### ⚠️ High Risk of Burnout")
                st.markdown(
                    f"Our assessment indicates you have a **{probability:.1%} chance** of experiencing burnout soon if no changes are made.")
            elif risk_level == 'medium':
                st.warning("### ⚠️ Medium Risk of Burnout")
                st.markdown(
                    f"Our assessment indicates you have a **{probability:.1%} chance** of developing burnout symptoms if stress continues.")
            else:
                st.success("### ✅ Low Risk of Burnout")
                st.markdown(
                    f"Our assessment indicates you have good balance with only a **{probability:.1%} chance** of burnout.")

        with col2:
            fig, ax = plt.subplots(figsize=(4, 1))
            meter_pos = {'low': (0, 30), 'medium': (31, 60), 'high': (61, 100)}
            ax.barh(0, 100, height=0.5, color='lightgrey', zorder=1)

            ax.barh(0, 33, height=0.5, color='green', zorder=2)
            ax.barh(0, 66, height=0.5, left=33, color='orange', zorder=2)
            ax.barh(0, 34, height=0.5, left=66, color='red', zorder=2)
            ax.scatter([meter_pos[risk_level]], [0], color='black', s=100, zorder=3)

            ax.plot([meter_pos[risk_level], meter_pos[risk_level]], [0, 0.5], color='black', linewidth=2, zorder=3)

            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.axis('off')

            st.pyplot(fig)

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

        if 'emotional_support' in user_data and user_data['emotional_support'].iloc[0] < 4:
            risk_factors.append(f"• Limited social support ({user_data['emotional_support'].iloc[0]}/10)")

        if 'exercise_hours' in user_data and user_data['exercise_hours'].iloc[0] < 2:
            risk_factors.append(f"• Limited physical activity ({user_data['exercise_hours'].iloc[0]} hours weekly)")

        if 'screen_time' in user_data and user_data['screen_time'].iloc[0] > 8:
            risk_factors.append(f"• High screen time ({user_data['screen_time'].iloc[0]} hours daily)")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(factor)
        else:
            st.markdown("• No significant risk factors identified")

        recommendations = backend.generate_recommendations(risk_level, user_data)

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
    st.markdown("Please answer the following questions honestly")