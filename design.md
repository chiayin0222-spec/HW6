prompt_v1:
  title: "Kaggle 50 Startups CRISP-DM sklearn Project"
  version: "v1"
  role: "You are a professional data scientist and machine learning instructor."

  objective: >
    Write a complete Scikit-learn solution for the Kaggle 50 Startups dataset.
    The solution must strictly follow the CRISP-DM process and include clear
    explanations, clean Python code, and expert-level feature analysis.

  dataset:
    name: "Kaggle 50 Startups"
    file_name: "50_Startups.csv"
    target_column: "Profit"
    features:
      - "R&D Spend"
      - "Administration"
      - "Marketing Spend"
      - "State"

  problem_type:
    learning_type: "Supervised Learning"
    task_type: "Regression"
    target: "Predict startup Profit"

  crisp_dm_steps:
    step_1_business_understanding:
      requirements:
        - "Explain the business problem."
        - "Explain why predicting Profit is useful."
        - "Explain that startup resources are limited."
        - "Explain how the model may help founders, investors, and analysts."
        - "Clearly state that this is a regression problem."
      expert_notes:
        - "Do not overclaim causality."
        - "The dataset has only 50 rows."
        - "Use association and prediction language, not causal language."

    step_2_data_understanding:
      requirements:
        - "Load the dataset using pandas."
        - "Show dataset shape."
        - "Show first five rows."
        - "Show data types."
        - "Check missing values."
        - "Check duplicate rows."
        - "Show descriptive statistics."
        - "Show State distribution."
        - "Show correlation matrix for numerical columns."
        - "Analyze Profit by State using groupby."
      code_checks:
        - "df.shape"
        - "df.head()"
        - "df.info()"
        - "df.describe()"
        - "df.isnull().sum()"
        - "df.duplicated().sum()"
        - "df['State'].value_counts()"
        - "df.corr(numeric_only=True)"
        - "df.groupby('State')['Profit'].agg(['count', 'mean', 'min', 'max', 'std'])"

    step_3_data_preparation:
      requirements:
        - "Separate X and y."
        - "Use Profit as target variable."
        - "Use R&D Spend, Administration, Marketing Spend, and State as features."
        - "Use OneHotEncoder for State."
        - "Do not use Label Encoding for State."
        - "Use ColumnTransformer."
        - "Use sklearn Pipeline."
        - "Use train_test_split with test_size=0.2 and random_state=42."
      preprocessing:
        numerical_features:
          - "R&D Spend"
          - "Administration"
          - "Marketing Spend"
        categorical_features:
          - "State"
        categorical_encoding:
          method: "OneHotEncoder"
          parameters:
            drop: "first"
            handle_unknown: "ignore"

    step_4_modeling:
      algorithm:
        primary_model: "LinearRegression"
        library: "scikit-learn"
      model_experiments:
        model_1:
          name: "R&D Only"
          features:
            - "R&D Spend"
          purpose: "Check the predictive power of the core feature."
        model_2:
          name: "R&D + Marketing"
          features:
            - "R&D Spend"
            - "Marketing Spend"
          purpose: "Check whether Marketing adds value beyond R&D."
        model_3:
          name: "Numerical Features"
          features:
            - "R&D Spend"
            - "Marketing Spend"
            - "Administration"
          purpose: "Check whether Administration improves prediction."
        model_4:
          name: "All Features"
          features:
            - "R&D Spend"
            - "Marketing Spend"
            - "Administration"
            - "State"
          purpose: "Check whether State improves prediction."

    step_5_evaluation:
      requirements:
        - "Evaluate each model with train-test split."
        - "Evaluate each model with 5-fold cross-validation."
        - "Print a comparison table."
        - "Select the final model based on CV R2 Mean."
      metrics:
        - "R2 Score"
        - "MAE"
        - "RMSE"
        - "CV R2 Mean"
        - "CV R2 Std"
        - "CV RMSE Mean"
        - "CV RMSE Std"
      cross_validation:
        method: "KFold"
        parameters:
          n_splits: 5
          shuffle: true
          random_state: 42
      model_selection_rule:
        - "Prefer the model with the best CV R2 Mean."
        - "If models are close, prefer the simpler and more interpretable model."
        - "Explain whether State adds meaningful value."
        - "Explain whether R&D Spend dominates the prediction."

    step_6_deployment:
      requirements:
        - "Create a deployment simulation."
        - "Predict Profit for a new startup."
        - "Save the final model with joblib."
        - "Filename should be startup_profit_model_v1.pkl."
      sample_input:
        R&D Spend: 120000
        Administration: 130000
        Marketing Spend: 250000
        State: "New York"

  feature_analysis:
    R&D Spend:
      role: "Core innovation factor"
      expected_importance: "High"
      recommendation: "Always keep"
      interpretation: >
        R&D Spend is expected to be the strongest predictor because it reflects
        product development, innovation ability, and technical competitiveness.

    Marketing Spend:
      role: "Market expansion factor"
      expected_importance: "Medium to high"
      recommendation: "Keep, but check correlation with R&D Spend"
      interpretation: >
        Marketing Spend may help prediction, but it may also overlap with company
        size and R&D Spend. Avoid overinterpreting it as an independent causal factor.

    Administration:
      role: "Operating cost and company scale factor"
      expected_importance: "Low to medium"
      recommendation: "Keep first, evaluate later"
      interpretation: >
        Administration may be weaker because it does not directly create revenue,
        but it may still reflect company scale and operational maturity.

    State:
      role: "Regional auxiliary factor"
      expected_importance: "Low to medium"
      recommendation: "Use One-Hot Encoding and avoid overinterpretation"
      interpretation: >
        State may reflect regional business environment, but the dataset is small,
        so it should be treated as an auxiliary variable only.

  code_requirements:
    language: "Python"
    libraries:
      - "pandas"
      - "numpy"
      - "joblib"
      - "scikit-learn"
    sklearn_modules:
      - "train_test_split"
      - "KFold"
      - "cross_val_score"
      - "ColumnTransformer"
      - "OneHotEncoder"
      - "Pipeline"
      - "LinearRegression"
      - "r2_score"
      - "mean_absolute_error"
      - "mean_squared_error"
    file_structure:
      output_file_name: "solve_50_startups_crispdm_v1.py"
      style:
        - "Use clear section comments."
        - "Separate helper functions."
        - "Use main() function."
        - "Use if __name__ == '__main__'."
        - "Print readable outputs."
        - "Handle missing file error."
        - "Check required columns before modeling."

  required_functions:
    - name: "build_pipeline"
      purpose: "Build sklearn preprocessing and regression pipeline."
    - name: "evaluate_train_test"
      purpose: "Evaluate model with R2, MAE, and RMSE on test set."
    - name: "evaluate_cross_validation"
      purpose: "Evaluate model with 5-fold CV using R2 and RMSE."
    - name: "data_understanding"
      purpose: "Print dataset overview and exploratory checks."
    - name: "run_model_experiments"
      purpose: "Train and evaluate four feature-set models."
    - name: "select_final_model"
      purpose: "Select best model based on CV R2 Mean."
    - name: "deployment_simulation"
      purpose: "Predict Profit for a new startup."
    - name: "save_model"
      purpose: "Save final pipeline using joblib."
    - name: "main"
      purpose: "Run the complete CRISP-DM workflow."

  output_requirements:
    must_include:
      - "Complete Python code."
      - "CRISP-DM comments."
      - "Business understanding explanation."
      - "Data understanding outputs."
      - "Four model experiments."
      - "Evaluation table."
      - "Expert interpretation."
      - "Deployment simulation."
      - "Saved model."
    output_format:
      - "Return the final Python code in one code block."
      - "Do not skip any CRISP-DM step."
      - "Make the code runnable as a single script."

  final_interpretation_template: >
    The model comparison shows that R&D Spend is expected to be the most important
    predictor of Profit. Marketing Spend may provide additional predictive value,
    while Administration may be weaker but still useful as a scale-related factor.
    State is treated as an auxiliary categorical feature and encoded using
    One-Hot Encoding. Because the dataset contains only 50 observations, the
    results should be interpreted as predictive associations rather than causal
    conclusions.

  constraints:
    - "Do not use Label Encoding for State."
    - "Do not claim causality."
    - "Do not remove Administration too early."
    - "Do not rely only on one train-test split."
    - "Use cross-validation because the dataset is small."
    - "Keep the solution beginner-friendly but professionally structured."