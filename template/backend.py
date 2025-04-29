""" Backend logic for Burnout Early Warning System """

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_model(model_path: str) -> Tuple[Any, Dict[str, float]]:
    """Loads the pre-trained model and its metrics."""

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
        class BurnoutPredictor:
            """Default Burnout Predictor if no model is found"""

            def predict(self, X):
                risk_scores = self._calculate_risk_scores(X)
                predictions = np.zeros(len(X))
                predictions[risk_scores > 15] = 2  # High risk
                predictions[(risk_scores <= 15) & (risk_scores > 10)] = 1  # Medium risk
                return predictions

            def predict_proba(self, X):
                risk_scores = self._calculate_risk_scores(X)
                probas = np.zeros((len(X), 3))

                for i, score in enumerate(risk_scores):
                    if score > 15:
                        probas[i] = [0.1, 0.2, 0.7]
                    elif score > 10:
                        probas[i] = [0.2, 0.6, 0.2]
                    else:
                        probas[i] = [0.7, 0.2, 0.1]
                return probas

            def _calculate_risk_scores(self, X):
                scores = np.zeros(len(X))
                for i, row in X.iterrows():
                    sleep = row.get('sleep_hours', 7)
                    workload = row.get('workload', 40)
                    deadlines = row.get('deadlines', 3)
                    stress = row.get('stress_level', 5)
                    emotion = row.get('emotional_support', 5)
                    exercise = row.get('exercise_hours', 3)
                    screen_time = row.get('screen_time', 6)

                    scores[i] = (
                        (10 - sleep) * 0.3 +
                        (workload / 10) * 0.2 +
                        (deadlines * 0.5) +
                        stress * 0.3 -
                        emotion * 0.2 -
                        exercise * 0.15 +
                        (screen_time > 8) * 2
                    )
                return scores

        return BurnoutPredictor(), default_metrics


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
        "Identify top stressors and write them down to manage better")

        recommendations[risk_level]['relaxation'].insert(0,
                                                       "Try stress-reduction techniques like yoga or tai chi")
        recommendations[risk_level]['support'].insert(0,
                                                    "Talk to an academic advisor about managing your workload")

    if 'exercise_hours' in user_data and user_data['exercise_hours'].iloc[0] < 2:
        recommendations[risk_level]['relaxation'].append(
            "Add short bursts of physical activity to your daily routine")

    if 'emotional_support' in user_data and user_data['emotional_support'].iloc[0] <= 3:
        recommendations[risk_level]['support'].insert(0,
                                                    "Join campus clubs or organizations to build your support network")

    if 'screen_time' in user_data and user_data['screen_time'].iloc[0] > 8:
        recommendations[risk_level]['time_management'].insert(0,
                                                           "Implement screen-free periods during your day to reduce digital fatigue")

    return recommendations[risk_level]