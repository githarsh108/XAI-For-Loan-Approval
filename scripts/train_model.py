import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_model():
    print("Loading data...")
    df = pd.read_csv(os.path.join('data', 'loan_data.csv'))
    
    # 1. Preprocessing
    print("Preprocessing data...")
    
    # Drop Loan_ID
    df = df.drop('Loan_ID', axis=1)
    
    # Handle missing values
    # Categorical: Fill with mode
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Numerical: Fill with median
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # 2. Encoding categorical variables
    label_encoders = {}
    for col in cat_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            
    # Target encoding
    le_target = LabelEncoder()
    df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])
    label_encoders['Loan_Status'] = le_target
    
    # 3. Feature Selection
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    # Keep track of feature names
    feature_names = X.columns.tolist()
    
    # 4. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train Model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save model and preprocessors
    print("\nSaving artifacts...")
    artifacts = {
        'model': rf_model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names
    }
    
    joblib.dump(artifacts, os.path.join('models', 'loan_model_assets.pkl'))
    print("Model and assets saved to models/loan_model_assets.pkl")

if __name__ == "__main__":
    train_model()
