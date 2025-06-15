Example Usage (for testing purposes, assumes 'data/raw/' exists with sample CSVs)
To run this, you'd need to have dummy CSV files like 'application_train.csv', 'bureau.csv'
in a 'data/raw' directory, with some columns like 'SK_ID_CURR', 'AMT_INCOME_TOTAL', 'TARGET'.

Example: Create dummy files for testing
os.makedirs('data/raw', exist_ok=True)
pd.DataFrame({'SK_ID_CURR': [1, 2, 3], 'AMT_INCOME_TOTAL': [10000, 50000, 120000], 'TARGET': [0, 1, 0]}).to_csv('data/raw/application_train.csv', index=False)
pd.DataFrame({'SK_ID_CURR': [1, 2], 'CREDIT_ACTIVE': ['Active', 'Closed'], 'AMT_CREDIT_SUM': [1000, 5000]}).to_csv('data/raw/bureau.csv', index=False)


if __name__ == "__main__":
    # Backlog Item 1: Simple
    backlog_item_1 = """
    ETL for applicant data. We need to join application_train with bureau on SK_ID_CURR using a left join.
    Impute missing AMT_ANNUITY values with the mean.
    The main goal is to predict loan repayment likelihood, so apply a generic ML model with TARGET as the target column.
    Output to a dataframe.
    """
    print("--- Parsing Backlog Item 1 ---")
    parsed_result_1 = parse_etl_backlog_item(backlog_item_1)
    if isinstance(parsed_result_1, dict):
        import json
        print(json.dumps(parsed_result_1, indent=2))
    else:
        print(parsed_result_1)

    # Backlog Item 2: More complex, with feature engineering and specific output
    backlog_item_2 = """
    Create a pipeline for enhanced risk assessment.
    Initial data from application_train and previous_application.
    Join application_train with previous_application on SK_ID_CURR (inner join).
    Handle missing values in AMT_CREDIT by mean imputation.
    Calculate a new feature: 'DEBT_INCOME_RATIO' from AMT_CREDIT and AMT_INCOME_TOTAL.
    Model objective: classify loan default using an XGBoost model. TARGET is the output.
    Save results as parquet to an S3 bucket at s3://homecredit-processed/risk_data.
    """
    print("\n--- Parsing Backlog Item 2 ---")
    parsed_result_2 = parse_etl_backlog_item(backlog_item_2)
    if isinstance(parsed_result_2, dict):
        import json
        print(json.dumps(parsed_result_2, indent=2))
    else:
        print(parsed_result_2)

    # Backlog Item 3: Missing table/column for demonstration of validation error
    backlog_item_3 = """
    Process client data from non_existent_table.csv.
    Join with application_train on SK_ID_CLIENT.
    Filter by invalid_column > 100.
    """
    print("\n--- Parsing Backlog Item 3 (Expecting Error) ---")
    parsed_result_3 = parse_etl_backlog_item(backlog_item_3)
    print(parsed_result_3)