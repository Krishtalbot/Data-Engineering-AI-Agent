import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import great_expectations as gx
from great_expectations.dataset.sparkdf_dataset import SparkDFDataset

# Constants
APPLICATION_TRAIN_PATH = "data/raw/application_train.csv"
BUREAU_PATH = "data/raw/bureau.csv"
OUTPUT_PATH = "data/output/loan_default_predictions.parquet"
JOIN_KEY = "SK_ID_CURR"
JOIN_TYPE = "inner"

def initialize_spark_session(app_name="LoanDefaultPrediction"):
    """Initializes and returns a Spark session."""
    return SparkSession.builder.appName(app_name).getOrCreate()

def initialize_gx_context():
    """Initializes and returns a Great Expectations context."""
    return gx.get_context()

def read_data(spark, path):
    """Reads a CSV file into a Spark DataFrame."""
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df
    except Exception as e:
        raise Exception(f"Error reading data from {path}: {e}")

def perform_join(application_train_df, bureau_df, join_key, join_type):
    """Performs a join operation between two DataFrames."""
    # Attempt a broadcast join if bureau_df is small enough
    if bureau_df.count() < 10000:  # Adjust threshold as needed
        joined_df = application_train_df.join(F.broadcast(bureau_df), on=join_key, how=join_type)
    else:
        joined_df = application_train_df.join(bureau_df, on=join_key, how=join_type)
    return joined_df

def define_and_validate_expectation_suite(context, suite_name, df, expectations):
    """Defines and validates a Great Expectations suite."""
    try:
        suite = context.get_expectation_suite(suite_name)
        print(f"Loaded ExpectationSuite `{suite.name}` containing {len(suite.expectations)} expectations.")
    except gx.exceptions.DataContextError:
        suite = context.create_expectation_suite(suite_name)

    for check in expectations:
        expectation_type = check["expectation_type"]
        column = check.get("column")
        kwargs = check.get("kwargs", {})

        if expectation_type == "expect_column_values_to_not_be_null":
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBePresent(column=column))
        elif expectation_type == "expect_column_pair_values_to_be_unique":
            suite.add_expectation(gx.expectations.ExpectColumnPairValuesToBeUnique(column_A=column, column_B=column, ignore_row_if=kwargs.get("ignore_row_if", "any")))
        elif expectation_type == "expect_column_values_to_be_of_type":
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column=column, type_=kwargs["type"]))
        elif expectation_type == "expect_column_values_to_be_between":
            suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column=column, min_value=kwargs["min_value"], max_value=kwargs["max_value"]))
        else:
            raise ValueError(f"Unsupported expectation type: {expectation_type}")

    context.save_expectation_suite(suite)
    ge_df = SparkDFDataset(df)
    validation_results = ge_df.validate(expectation_suite=suite)

    if not validation_results["success"]:
        print(f"Data quality checks failed for {suite_name}!")
        for result in validation_results["results"]:
            if not result["success"]:
                print(f"  Failure: {result['expectation_config']['expectation_type']}")
                print(f"    - Column: {result['expectation_config'].get('kwargs', {}).get('column')}")
                print(f"    - Details: {result['result']}")
                severity = next((check["severity"] for check in expectations if check["expectation_type"] == result['expectation_config']['expectation_type']), None)
                if severity == "CRITICAL":
                    raise Exception(f"Critical data quality check failed for {suite_name}.")
                elif severity == "WARNING":
                    print("Warning: Data quality issue detected.")
    context.build_data_docs()

def handle_missing_values(df):
    """Imputes missing values in numeric columns with the mean and categorical columns with the mode."""
    cleaned_data = df
    # Impute numeric columns with mean
    for col in cleaned_data.columns:
        if isinstance(cleaned_data.schema[col].dataType, (IntegerType, DoubleType, FloatType)):
            mean_val = cleaned_data.select(F.mean(col)).first()[0]
            if mean_val is not None:
                cleaned_data = cleaned_data.fillna({col: mean_val})

    # Impute categorical columns with mode
    for col in cleaned_data.columns:
        if not isinstance(cleaned_data.schema[col].dataType, (IntegerType, DoubleType, FloatType)):
            mode_val = cleaned_data.groupBy(col).count().orderBy(F.desc("count")).first()[col]
            if mode_val is not None:
                cleaned_data = cleaned_data.fillna({col: mode_val})
    return cleaned_data

def apply_loan_scoring_model(df):
    """Applies a placeholder loan scoring model. Replace with actual model application."""
    scored_data = df.withColumn("LOAN_REPAYMENT_SCORE", F.lit(None).cast(DoubleType()))
    return scored_data

def write_output(df, path):
    """Writes the DataFrame to a Parquet file."""
    try:
        df.write.parquet(path, mode="overwrite")
        print(f"Data written to {path}")
    except Exception as e:
        raise Exception(f"Error writing data to {path}: {e}")

def main():
    """Main function to execute the loan default prediction pipeline."""
    # Initialize Spark session
    spark = initialize_spark_session()

    # Initialize Great Expectations context
    context = initialize_gx_context()

    # Read source data
    application_train_df = read_data(spark, APPLICATION_TRAIN_PATH)
    bureau_df = read_data(spark, BUREAU_PATH)

    # Join tables
    joined_df = perform_join(application_train_df, bureau_df, JOIN_KEY, JOIN_TYPE)

    # Cache the joined DataFrame before validation and cleaning
    joined_df = joined_df.cache()

    # Data Quality Checks - Post Join
    data_quality_checks_joined = [
        {"expectation_type": "expect_column_values_to_not_be_null", "column": "SK_ID_CURR", "severity": "CRITICAL", "description": "Ensure no null values in join key after join."},
        {"expectation_type": "expect_column_pair_values_to_be_unique", "column": "SK_ID_CURR", "kwargs": {"ignore_row_if": "any_value_is_missing"}, "severity": "CRITICAL", "description": "Ensure SK_ID_CURR is unique after the join."}
    ]
    define_and_validate_expectation_suite(context, "joined_df_suite", joined_df, data_quality_checks_joined)

    # Handle missing values
    cleaned_data = handle_missing_values(joined_df)

    # Cache the cleaned DataFrame before validation and scoring
    cleaned_data = cleaned_data.cache()

    # Data Quality Checks - Post Cleaning
    data_quality_checks_cleaned = [
        {"expectation_type": "expect_column_values_to_not_be_null", "column": "AMT_INCOME_TOTAL", "severity": "WARNING", "description": "Check for remaining nulls in AMT_INCOME_TOTAL after imputation."},
        {"expectation_type": "expect_column_values_to_be_of_type", "column": "AMT_INCOME_TOTAL", "kwargs": {"type": "numeric"}, "severity": "WARNING", "description": "Ensure AMT_INCOME_TOTAL is numeric after imputation."}
    ]
    define_and_validate_expectation_suite(context, "cleaned_data_suite", cleaned_data, data_quality_checks_cleaned)

    # Apply loan default scoring model
    scored_data = apply_loan_scoring_model(cleaned_data)

    # Cache the scored DataFrame before validation and writing
    scored_data = scored_data.cache()

    # Output: Apply data quality checks and write output
    data_quality_checks_scoring = [
        {"expectation_type": "expect_column_values_to_not_be_null", "column": "LOAN_REPAYMENT_SCORE", "severity": "CRITICAL", "description": "Ensure the final score is never null."},
        {"expectation_type": "expect_column_values_to_be_between", "column": "LOAN_REPAYMENT_SCORE", "kwargs": {"min_value": 0, "max_value": 100}, "severity": "CRITICAL", "description": "Ensure the final score is between 0 and 100."},
        {"expectation_type": "expect_column_values_to_be_of_type", "column": "LOAN_REPAYMENT_SCORE", "kwargs": {"type": "numeric"}, "severity": "CRITICAL", "description": "Ensure the final score is numeric."}
    ]
    define_and_validate_expectation_suite(context, "loan_default_predictions_suite", scored_data, data_quality_checks_scoring)

    # Write the final DataFrame to a Parquet file
    write_output(scored_data, OUTPUT_PATH)

    # Stop the Spark session
    spark.stop()

if __name__ == "__main__":
    main()