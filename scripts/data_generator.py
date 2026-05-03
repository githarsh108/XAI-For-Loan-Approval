import pandas as pd
import numpy as np
import os

def generate_loan_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate features
    data = {
        'Loan_ID': [f'LP{i:06d}' for i in range(n_samples)],
        'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.8, 0.2]),
        'Married': np.random.choice(['Yes', 'No'], n_samples, p=[0.65, 0.35]),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples, p=[0.57, 0.17, 0.16, 0.10]),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.78, 0.22]),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples, p=[0.14, 0.86]),
        'ApplicantIncome': np.random.randint(2500, 20000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 10000, n_samples),
        'LoanAmount': np.random.randint(50, 500, n_samples),
        'Loan_Amount_Term': np.random.choice([120, 180, 240, 360, 480], n_samples, p=[0.01, 0.07, 0.02, 0.85, 0.05]),
        'Credit_History': np.random.choice([1.0, 0.0], n_samples, p=[0.84, 0.16]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n_samples, p=[0.33, 0.38, 0.29]),
    }
    
    df = pd.DataFrame(data)
    
    # Logic for Loan_Status (making it semi-realistic)
    # Success probability based on Credit_History, Income, and LoanAmount
    # Probability = 0.7 * Credit_History + 0.2 * (Income/10000) - 0.1 * (LoanAmount/500)
    
    prob = (0.6 * df['Credit_History'] + 
            0.2 * (df['ApplicantIncome'] / 20000) + 
            0.2 * (df['Education'] == 'Graduate').astype(int) -
            0.1 * (df['LoanAmount'] / 500))
    
    # Add some noise
    prob += np.random.normal(0, 0.1, n_samples)
    
    df['Loan_Status'] = (prob > 0.4).map({True: 'Y', False: 'N'})
    
    # Introduce some missing values to make preprocessing realistic
    for col in ['Gender', 'Married', 'LoanAmount', 'Credit_History']:
        mask = np.random.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan
        
    return df

if __name__ == "__main__":
    df = generate_loan_data()
    output_path = os.path.join('data', 'loan_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {output_path}")
    print(df.head())
