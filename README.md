# Burnout Early Warning System

## Overview

A diagnostic application for evaluating burnout risk based on user metrics such as sleep, workload, stress, and lifestyle factors. The system outputs a predicted risk classification and an actionable set of recommendations. 

An integrated fallback heuristic ensures continuous availability if the primary predictive model cannot be loaded.

## Key Features

- Input pipeline for physiological, behavioral, and routine metrics.
- Probability-based risk prediction for burnout states.
- Interactive user interface built with Streamlit for data entry and reviewing results.
- Auto-generated intervention strategies tied to specific risk profiles.
- Automated fallback classifier for environments lacking trained model files.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Matplotlib

## Execution

1. Clone the repository and navigate to the project directory.
2. Install the necessary dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib joblib scikit-learn
   ```
3. Run the application:
   ```bash
   python -m streamlit run code.py
   ```

## Workflow

1. Form Submission: Complete the diagnostic questionnaire covering sleep, workload, interactions, and general mood.
2. Evaluation: System runs inference using the predefined model or fallback heuristic.
3. Reporting: The interface presents the predicted risk level alongside specific, module-based recommendations.
