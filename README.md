# Explainable AI for Loan Approval Prediction

This project demonstrates how to use Machine Learning for loan approval prediction and how to explain the model's decisions using SHAP (SHapley Additive exPlanations).

## Project Structure
- `data/`: Contains the generated loan dataset.
- `models/`: Stores the trained model and preprocessing artifacts.
- `scripts/`:
  - `data_generator.py`: Generates synthetic, realistic loan data.
  - `train_model.py`: Preprocesses data, trains a Random Forest, and saves assets.
- `app/`:
  - `main.py`: Interactive Streamlit application with XAI visualizations.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data (Optional - CSV is already included)
```bash
python scripts/data_generator.py
```

### 3. Train the Model
```bash
python scripts/train_model.py
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app/main.py
```

## Key Technologies
- **Scikit-learn**: For Random Forest Classification.
- **SHAP**: For model interpretability (Explainable AI).
- **Streamlit**: For the interactive web interface.
- **Pandas/Numpy**: For data manipulation.
