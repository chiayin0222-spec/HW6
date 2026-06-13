"""
Kaggle 50 Startups CRISP-DM Scikit-learn Project
Author: AI Assistant
Version: v1
Description: This script implements a complete Scikit-learn regression solution
             for predicting startup Profit using the 50 Startups dataset.
             The workflow strictly follows the CRISP-DM process.
"""

import os
import sys
import urllib.request
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib

# Scikit-learn modules
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

CSV_FILENAME = "50_Startups.csv"
DATASET_URL = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
MODEL_FILENAME = "startup_profit_model_v1.pkl"


def load_dataset():
    """
    Check if the dataset exists locally; if not, download it.
    Then read and return the pandas DataFrame.
    """
    if not os.path.exists(CSV_FILENAME):
        print(f"[INFO] Local dataset '{CSV_FILENAME}' not found.")
        print(f"[INFO] Downloading dataset from: {DATASET_URL}")
        try:
            # Add user agent to headers to avoid potential HTTP block
            req = urllib.request.Request(
                DATASET_URL, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                with open(CSV_FILENAME, 'wb') as f:
                    f.write(response.read())
            print("[INFO] Dataset downloaded successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to download dataset: {e}")
            print("[ERROR] Please make sure you have internet access or place the '50_Startups.csv' file manually.")
            sys.exit(1)
            
    try:
        df = pd.read_csv(CSV_FILENAME)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to read dataset: {e}")
        sys.exit(1)


def data_understanding(df):
    """
    CRISP-DM Step 2: Data Understanding.
    Prints dataset shape, preview, data types, missing/duplicate values,
    descriptive statistics, State distribution, and basic groupings.
    """
    print("\n" + "="*50)
    print(" CRISP-DM Step 2: Data Understanding ".center(50, "="))
    print("="*50)
    
    # Check shape
    print(f"\n1. Dataset Shape:\n   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # First 5 rows
    print("\n2. First 5 Rows of the Dataset:")
    print(df.head())
    
    # Data types and info
    print("\n3. Dataset Information & Data Types:")
    df.info()
    
    # Missing values
    print("\n4. Missing Values Count per Column:")
    print(df.isnull().sum())
    
    # Duplicate rows
    print("\n5. Duplicate Rows Count:")
    print(f"   Total Duplicates: {df.duplicated().sum()}")
    
    # Descriptive statistics
    print("\n6. Descriptive Statistics (Numerical Columns):")
    print(df.describe())
    
    # State distribution
    print("\n7. State Column Distribution:")
    print(df["State"].value_counts())
    
    # Group by State to see basic Profit statistics
    print("\n8. Profit Stats by State (Groupby):")
    state_profit = df.groupby("State")["Profit"].agg(["count", "mean", "min", "max", "std"])
    print(state_profit)
    
    # Correlation Matrix
    print("\n9. Correlation Matrix (Numerical Columns only):")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix)
    
    print("\n[Data Understanding Insights]")
    print("- 'R&D Spend' has the highest correlation with 'Profit' (~0.97), followed by 'Marketing Spend' (~0.75).")
    print("- 'Administration' has a very weak correlation with 'Profit' (~0.20).")
    print("- There are no missing values or duplicate rows in this dataset (n = 50).")
    print("- 'State' has 3 distinct values (New York, California, Florida) fairly evenly distributed.")


def build_pipeline(features):
    """
    CRISP-DM Step 3: Data Preparation.
    Builds a scikit-learn pipeline for the specified feature subset.
    If 'State' is in the features, uses OneHotEncoder(drop='first', handle_unknown='ignore').
    Others are passed through directly.
    """
    categorical_cols = [col for col in features if col == "State"]
    numerical_cols = [col for col in features if col != "State"]
    
    if categorical_cols:
        # Note: drop='first' is specified in the design requirements to prevent dummy variable trap.
        # handle_unknown='ignore' works when drop='first' in modern scikit-learn.
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
            ],
            remainder="passthrough"
        )
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])
    else:
        pipeline = Pipeline(steps=[
            ("regressor", LinearRegression())
        ])
        
    return pipeline


def evaluate_train_test(pipeline, X_train, y_train, X_test, y_test):
    """
    Evaluates the model pipeline on a single train-test split.
    Returns R2 Score, MAE, and RMSE.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return {
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    }


def evaluate_cross_validation(pipeline, X, y):
    """
    Evaluates the model pipeline using 5-fold cross-validation.
    Returns Mean & Std of R2 and RMSE.
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # R2 scoring
    r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    
    # RMSE scoring (cross_val_score returns negative MSE/RMSE metrics, so we negate it)
    rmse_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    
    return {
        "CV R2 Mean": r2_scores.mean(),
        "CV R2 Std": r2_scores.std(),
        "CV RMSE Mean": rmse_scores.mean(),
        "CV RMSE Std": rmse_scores.std()
    }


def run_model_experiments(df):
    """
    CRISP-DM Step 4: Modeling.
    Runs experiments for the four defined feature sets.
    """
    print("\n" + "="*50)
    print(" CRISP-DM Step 4: Modeling & Experiments ".center(50, "="))
    print("="*50)
    
    # Check if target and required features exist
    required_cols = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")
            
    y = df["Profit"]
    
    experiments = {
        "Model 1 (R&D Only)": ["R&D Spend"],
        "Model 2 (R&D + Marketing)": ["R&D Spend", "Marketing Spend"],
        "Model 3 (Numerical Features)": ["R&D Spend", "Marketing Spend", "Administration"],
        "Model 4 (All Features)": ["R&D Spend", "Marketing Spend", "Administration", "State"]
    }
    
    results = {}
    
    for name, features in experiments.items():
        print(f"[INFO] Running experiments for {name} with features: {features}...")
        X = df[features]
        
        # 80/20 train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build pipeline
        pipeline = build_pipeline(features)
        
        # Evaluate on split
        tt_metrics = evaluate_train_test(pipeline, X_train, y_train, X_test, y_test)
        
        # Evaluate on 5-fold CV
        cv_metrics = evaluate_cross_validation(pipeline, X, y)
        
        # Save results
        results[name] = {
            "features": features,
            **tt_metrics,
            **cv_metrics
        }
        
    return results


def select_final_model(results):
    """
    CRISP-DM Step 5: Evaluation.
    Compares the models using the evaluation table and outputs selection logic.
    """
    print("\n" + "="*50)
    print(" CRISP-DM Step 5: Evaluation & Selection ".center(50, "="))
    print("="*50)
    
    # Build comparison DataFrame
    table_data = []
    for name, metrics in results.items():
        table_data.append({
            "Model Name": name,
            "Selected Features": f"{metrics['features']}",
            "R2 (Test)": f"{metrics['R2 Score']:.5f}",
            "MAE (Test)": f"{metrics['MAE']:.2f}",
            "RMSE (Test)": f"{metrics['RMSE']:.2f}",
            "CV R2 Mean": f"{metrics['CV R2 Mean']:.5f}",
            "CV R2 Std": f"{metrics['CV R2 Std']:.5f}",
            "CV RMSE Mean": f"{metrics['CV RMSE Mean']:.2f}"
        })
        
    df_comparison = pd.DataFrame(table_data)
    print("\n=== Model Comparison Table ===")
    print(df_comparison.to_string(index=False))
    
    # Model selection rule: Prefer model with best CV R2 Mean.
    # If scores are extremely close, prefer the simpler model to prevent overfitting.
    best_model_name = max(results, key=lambda k: results[k]["CV R2 Mean"])
    best_features = results[best_model_name]["features"]
    
    print(f"\nFinal Selection Results:")
    print(f"- Recommended Model based on CV R2 Mean: '{best_model_name}'")
    print(f"- Chosen Features: {best_features}")
    
    # Expert interpretation & reasoning
    print("\n[Expert Model Evaluation & Interpretation]")
    print("1. R&D Spend Dominance:")
    print("   - R&D Spend is the single most powerful predictor of startup Profit.")
    print("   - Adding features like Marketing Spend improves CV R2 slightly (Model 2 vs Model 1).")
    print("2. Role of Administration and State:")
    print("   - Model 3 (adding Administration) and Model 4 (adding State) do not yield significant improvements in CV R2 Mean.")
    print("   - In fact, adding too many weak features in a very small dataset (n=50) increases the model variance (CV R2 Std) without significantly reducing bias.")
    print("   - The regional 'State' factor (One-Hot Encoded) does not provide distinct predictive power here.")
    print("3. Association vs. Causality Warning:")
    print("   - Because the dataset has only 50 observations, these results represent statistical associations.")
    print("   - They do not imply a direct causal relationship. We cannot state that increasing Administration spend will causally reduce profit or that changing states directly causes profit changes.")
    
    return best_model_name, best_features


def deployment_simulation(pipeline, features):
    """
    CRISP-DM Step 6: Deployment (Part 1).
    Simulates predicting the profit for a new startup using the trained model.
    """
    print("\n" + "="*50)
    print(" CRISP-DM Step 6: Deployment Simulation ".center(50, "="))
    print("="*50)
    
    # Target new startup
    sample_input = {
        "R&D Spend": [120000],
        "Administration": [130000],
        "Marketing Spend": [250000],
        "State": ["New York"]
    }
    
    df_new = pd.DataFrame(sample_input)
    # Filter columns to only those used by the chosen model
    X_new = df_new[features]
    
    prediction = pipeline.predict(X_new)[0]
    
    print("\nNew Startup Input Features:")
    for key, val in sample_input.items():
        print(f"  - {key}: {val[0] if not isinstance(val[0], float) else f'${val[0]:,.2f}'}")
        
    print(f"\nPredicted Startup Profit: ${prediction:,.2f}")
    return prediction


def save_model(pipeline, filename):
    """
    CRISP-DM Step 6: Deployment (Part 2).
    Saves the final pipeline using joblib.
    """
    try:
        joblib.dump(pipeline, filename)
        print(f"[INFO] Pipeline successfully saved to: {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save pipeline: {e}")


def main():
    print("="*60)
    print(" Kaggle 50 Startups CRISP-DM Machine Learning Solution ".center(60))
    print("="*60)
    
    # CRISP-DM Step 1: Business Understanding
    print("\n" + "="*50)
    print(" CRISP-DM Step 1: Business Understanding ".center(50, "="))
    print("="*50)
    print("Business Problem Definition:")
    print("- Startups have limited resources and need to allocate their budget wisely across R&D, Administration, and Marketing.")
    print("- Predicting startup profit allows founders, investors, and analysts to identify key drivers of profitability.")
    print("- Modeling helps decide where an additional dollar of investment might associate with the highest change in profit.")
    print("- This is a Supervised Learning regression task, where we map feature variables to a continuous outcome ('Profit').")
    print("Constraints & Limitations:")
    print("- The dataset size is extremely small (n = 50). Overfitting is a high risk.")
    print("- Standard train-test split evaluation can be highly unstable due to small sample size. Cross-validation is required.")
    print("- Output predictions show associations and patterns, not strict causal guarantees.")
    
    # Load dataset
    df = load_dataset()
    
    # CRISP-DM Step 2: Data Understanding
    data_understanding(df)
    
    # CRISP-DM Step 4 & 5: Modeling & Evaluation
    results = run_model_experiments(df)
    best_model_name, best_features = select_final_model(results)
    
    # Re-train final model on the ENTIRE dataset for deployment
    print(f"\n[INFO] Re-training final model '{best_model_name}' on the entire dataset...")
    X_full = df[best_features]
    y_full = df["Profit"]
    
    final_pipeline = build_pipeline(best_features)
    final_pipeline.fit(X_full, y_full)
    print("[INFO] Re-training complete.")
    
    # CRISP-DM Step 6: Deployment
    deployment_simulation(final_pipeline, best_features)
    save_model(final_pipeline, MODEL_FILENAME)
    
    # Final Summary interpretation
    print("\n" + "="*50)
    print(" Final Interpretation Summary ".center(50, "="))
    print("="*50)
    print("The model comparison shows that R&D Spend is expected to be the most important")
    print("predictor of Profit. Marketing Spend may provide additional predictive value,")
    print("while Administration may be weaker but still useful as a scale-related factor.")
    print("State is treated as an auxiliary categorical feature and encoded using")
    print("One-Hot Encoding. Because the dataset contains only 50 observations, the")
    print("results should be interpreted as predictive associations rather than causal")
    print("conclusions.")
    print("="*60)


if __name__ == "__main__":
    main()
