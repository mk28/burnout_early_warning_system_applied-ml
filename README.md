# Burnout Early Warning System (BEWS)

## Overview

In today's highly competitive academic and professional landscapes, individuals frequently face mounting pressures from deadlines, assignments, and personal obligations. This chronic pressure often leads to burnout—a state of emotional, physical, and mental exhaustion caused by prolonged stress.

Unfortunately, burnout is frequently underdiagnosed and untreated until it results in severe clinical or academic consequences. Recognizing the early indicators of burnout enables individuals to take preventive action before reaching critical stages of exhaustion.

This project implements a Burnout Early Warning System (BEWS) that leverages predictive models to evaluate an individual's burnout risk based on behavioral, physiological, and routine parameters, offering timely interventions and personalized resilience recommendations.

## Features

- **Predictive Risk Modeling**: Analyzes multi-dimensional inputs (sleep quality, workload, stress, social isolation) to compute a probabilistic assessment of burnout risk.
- **Diagnostic Interface**: A clean, accessible Streamlit frontend for users to log their metrics and receive immediate feedback.
- **Actionable Interventions**: Generates personalized, targeted recommendations spanning relaxation, time management, and support mechanisms tailored to the specific risk profile.
- **Fallback Heuristics**: Ensures system availability by gracefully falling back to a deterministic heuristic model if the primary machine learning model is unavailable.

## Architecture and Stack

- **Frontend Application**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning Integration**: Scikit-Learn, Joblib
- **Visualization**: Matplotlib

## Installation and Usage

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd burnout_early_warning_system_applied-ml
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install streamlit pandas numpy matplotlib joblib scikit-learn
   ```

3. **Run the Application**
   Launch the diagnostic interface via Streamlit:
   ```bash
   streamlit run code.py
   ```

## Workflow

1. **Input**: Users provide data across several categories including sleep metrics, workload, stress context, social interaction, physiological habits, and emotional state.
2. **Analysis**: The system processes these inputs through the prediction engine.
3. **Output**: Displays a visual indicator of risk probability along with a custom action plan that can be exported for offline review.
