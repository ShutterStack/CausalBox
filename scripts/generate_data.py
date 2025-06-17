# scripts/generate_data.py
import numpy as np
import pandas as pd
import os

def generate_dataset(n_samples=1000):
    np.random.seed(42)
    study_hours = np.random.normal(10, 2, n_samples)
    tuition_hours = np.random.normal(5, 1, n_samples)
    parental_education = np.random.choice(['High', 'Medium', 'Low'], n_samples)
    school_type = np.random.choice(['Public', 'Private'], n_samples)
    exam_score = 50 + 2 * study_hours + 1.5 * tuition_hours + np.random.normal(0, 5, n_samples)

    df = pd.DataFrame({
        'StudyHours': study_hours,
        'TuitionHours': tuition_hours,
        'ParentalEducation': parental_education,
        'SchoolType': school_type,
        'FinalExamScore': exam_score
    })
    
    # Ensure data directory exists
    os.makedirs('../data', exist_ok=True)
    df.to_csv('../data/sample_dataset.csv', index=False)
    return df

if __name__ == "__main__":
    generate_dataset()
    print("Dataset generated and saved to ../data/sample_dataset.csv")
