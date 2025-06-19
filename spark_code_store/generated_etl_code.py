import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.sql.types import DoubleType, IntegerType, LongType, FloatType, StringType
import great_expectations as gx
from great_expectations.core import ExpectationSuite
from great_expectations.dataset.sparkdf_dataset import SparkDFDataset
import great_expectations_experimental as gxe
from great_expectations_experimental.expectations import (
    expect_column_values_to_not_be_null_and_column_to_not_be_empty,
)


# Constants for file paths and suite names
APPLICATION_TRAIN_PATH = "data/raw/application_train.csv"
BUREAU_PATH = "data/raw/bureau.csv"
OUTPUT_PATH = "data/output/loan_default_predictions.parquet"
JOINED_DF_SUITE_NAME = "joined_df_suite"
CLEANED_DATA_SUITE_NAME = "cleaned_data_suite"
LOAN_DEFAULT_PREDICTIONS_SUITE_NAME = "loan_default_predictions_suite"
CRITICAL_SEVERITY = "CRITICAL"


def create_spark_session(app_name="LoanDefaultPrediction"):
    """Initializes and returns a Spark session with optimized memory settings."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.executor.memory", "4g")  # Give each executor 4GB of RAM
        .config("spark.driver.memory", "2g")  # Give the driver 2GB of RAM
        .config("spark.memory.offHeap.enabled", "true")  # Enable off-heap memory
        .config(
            "spark.memory.offHeap.size", "1g"
        )  # Allocate 1GB for off-heap (e.g., for certain Tungsten operations)
        .config(
            "spark.sql.shuffle.partitions", "200"
        )  # Increase shuffle partitions if you have many cores/data
        .getOrCreate()
    )


def read_data(spark, path):
    """Reads a CSV file into a Spark DataFrame. Includes basic error handling."""
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        print(f"Successfully read data from {path}. Schema:")
        df.printSchema()
        return df
    except Exception as e:
        # Raise a more specific IOError for file reading issues
        raise IOError(f"Error reading file {path}: {e}")


def perform_join(app_df, bureau_df):
    """Performs an inner join of two DataFrames on 'SK_ID_CURR'."""
    # Ensure SK_ID_CURR columns are of the same type before joining if inferSchema causes mismatches
    # For robust joins, it's good practice to cast join keys to a common type if unsure.
    # Here, assuming they are compatible after inferSchema.
    joined_df = app_df.join(bureau_df, on="SK_ID_CURR", how="inner")
    print("\n--- Joined DataFrame Schema ---")
    joined_df.printSchema()
    return joined_df


def define_join_expectations(context, suite_name=JOINED_DF_SUITE_NAME):
    """Defines Great Expectations suite for the joined DataFrame, focusing on join key quality."""
    # Create an ExpectationSuite object
    suite = ExpectationSuite(expectation_suite_name=suite_name)

    # Expectations as per requirements:
    # 1. Ensure no null values in join key after join (SK_ID_CURR)
    suite.add_expectation(
        gxe.expectations.expect_column_values_to_not_be_null_and_column_to_not_be_empty(  # Reverted to short name via gx.expectations
            column="SK_ID_CURR",
            meta={
                "severity": CRITICAL_SEVERITY,
                "description": "Ensure no null values in join key after join.",
            },
        )
    )
    # 2. Ensure SK_ID_CURR is unique after the join.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(  # Reverted to short name via gx.expectations
            column="SK_ID_CURR",
            meta={
                "severity": CRITICAL_SEVERITY,
                "description": "Ensure SK_ID_CURR is unique after the join.",
            },
        )
    )
    # Adding a type check for the join key, common for robustness
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(  # Reverted to short name via gx.expectations
            column="SK_ID_CURR",
            type="IntegerType",  # Using PySpark type string
            meta={
                "severity": "WARNING",
                "description": "Ensure SK_ID_CURR is integer type.",
            },
        )
    )

    # Add the created ExpectationSuite object to the DataContext
    context.add_expectation_suite(suite)
    # Save the expectation suite to the context's store
    context.save_expectation_suite(suite)
    print(f"Defined and saved ExpectationSuite: {suite_name}")
    return suite


def validate_data(context, df, suite_name):
    """
    Validates a DataFrame against a Great Expectations suite.
    Handles critical failures by raising an exception.
    """
    try:
        # Retrieve the suite by name
        suite = context.get_expectation_suite(suite_name)
    except gx.exceptions.DataContextError:
        print(
            f"Error: Expectation Suite '{suite_name}' not found. Cannot perform validation."
        )
        return False  # Indicate validation failure

    # Create a SparkDFDataset batch from the DataFrame
    batch = SparkDFDataset(df)
    results = batch.validate(expectation_suite=suite)

    if not results["success"]:
        print(f"\n!!! Data quality checks FAILED for suite: {suite_name} !!!")
        for result in results["results"]:
            if not result["success"]:
                expectation_type = result["expectation_config"]["expectation_type"]
                column_name = result["expectation_config"].get("column", "N/A")
                severity = result["expectation_config"]["meta"].get(
                    "severity", "WARNING"
                )
                description = result["expectation_config"]["meta"].get(
                    "description", "No description provided."
                )

                print(
                    f"  FAILED Expectation: {expectation_type} on column '{column_name}'"
                )
                print(f"    Description: {description}")
                print(f"    Result Details: {result['result']}")
                print(f"    Severity: {severity}")

                if severity == CRITICAL_SEVERITY:
                    context.build_data_docs()  # Build docs even on critical failure
                    raise Exception(
                        f"Critical data quality check failed during {suite_name}. Halting pipeline."
                    )
        # If any non-critical failures occurred, build data docs and continue (but return False for overall success)
        context.build_data_docs()
        return False
    else:
        print(f"\n--- Data quality checks PASSED for suite: {suite_name} ---")
        context.build_data_docs()
        return True


def impute_missing_values(df):
    """
    Imputes missing values in numeric columns with the mean and categorical columns with the mode.
    Handles various numeric types.
    """
    cleaned_data = df

    # Define common numeric types
    numeric_types = (DoubleType, IntegerType, LongType, FloatType)
    numeric_cols = [
        f.name
        for f in cleaned_data.schema.fields
        if isinstance(f.dataType, numeric_types)
    ]
    categorical_cols = [
        f.name
        for f in cleaned_data.schema.fields
        if isinstance(f.dataType, StringType)  # Check for StringType for categorical
    ]

    print("\n--- Imputing Missing Numeric Values ---")
    # Impute numeric columns with mean
    for col_name in numeric_cols:
        # Calculate mean, handling cases where column might be all null
        mean_value_row = cleaned_data.select(mean(col_name)).first()
        if mean_value_row is not None and mean_value_row[0] is not None:
            mean_value = mean_value_row[0]
            cleaned_data = cleaned_data.fillna({col_name: mean_value})
            print(f"  Imputed numeric column '{col_name}' with mean: {mean_value}")
        else:
            print(
                f"  Warning: Cannot impute mean for '{col_name}', column is all null or empty. Skipping."
            )

    print("\n--- Imputing Missing Categorical Values ---")
    # Impute categorical columns with mode
    for col_name in categorical_cols:
        # Calculate mode
        try:
            mode_value_row = (
                cleaned_data.groupBy(col_name)
                .count()
                .orderBy(col("count").desc())
                .first()
            )
            if mode_value_row is not None:
                mode_value = mode_value_row[col_name]
                cleaned_data = cleaned_data.fillna({col_name: mode_value})
                print(
                    f"  Imputed categorical column '{col_name}' with mode: '{mode_value}'"
                )
            else:
                print(
                    f"  Warning: Cannot impute mode for '{col_name}', column is empty or has no distinct values. Skipping."
                )
        except Exception as e:
            print(
                f"  Error calculating mode for '{col_name}': {e}. Skipping imputation for this column."
            )

    print("\n--- Cleaned Data Schema ---")
    cleaned_data.printSchema()
    return cleaned_data


def define_imputation_expectations(context, suite_name=CLEANED_DATA_SUITE_NAME):
    """Defines Great Expectations suite for data after imputation."""
    suite = ExpectationSuite(expectation_suite_name=suite_name)

    # Expectations as per requirements:
    # 1. Check for remaining nulls in AMT_INCOME_TOTAL after imputation (WARNING)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(  # Reverted to short name via gx.expectations
            column="AMT_INCOME_TOTAL",
            meta={
                "severity": "WARNING",
                "description": "Check for remaining nulls in AMT_INCOME_TOTAL after imputation.",
            },
        )
    )
    # 2. Ensure AMT_INCOME_TOTAL is numeric after imputation (WARNING)
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(  # Reverted to short name via gx.expectations
            column="AMT_INCOME_TOTAL",
            type="float",  # More robust check for general numeric types
            meta={
                "severity": "WARNING",
                "description": "Ensure AMT_INCOME_TOTAL is numeric after imputation.",
            },
        )
    )
    context.add_expectation_suite(suite)
    context.save_expectation_suite(suite)
    print(f"Defined and saved ExpectationSuite: {suite_name}")
    return suite


def apply_loan_scoring_model(df):
    """
    Placeholder for applying the loan default scoring model.
    Replace with actual model integration (e.g., Spark MLlib pipeline, UDF for external model).
    This example adds a dummy score based on existing data.
    """
    print("\n--- Applying Loan Scoring Model (Placeholder) ---")
    # For demonstration, assign a dummy score between 0-100.
    # In a real scenario, this would involve loading and applying a pre-trained model.
    # Example: scored_data = your_automl_model.transform(df)
    # Make sure 'AMT_INCOME_TOTAL' exists and is numeric for this dummy calculation.
    if "AMT_INCOME_TOTAL" in df.columns:
        scored_data = df.withColumn(
            "LOAN_REPAYMENT_SCORE",
            (col("AMT_INCOME_TOTAL") % 100).cast(
                DoubleType()
            ),  # Cast to DoubleType for consistency
        )
        print("  Dummy LOAN_REPAYMENT_SCORE column added based on AMT_INCOME_TOTAL.")
    else:
        # Fallback if AMT_INCOME_TOTAL is missing, or for a more generic placeholder
        scored_data = df.withColumn(
            "LOAN_REPAYMENT_SCORE",
            (pyspark.sql.functions.rand() * 100).cast(DoubleType()),  # Random score
        )
        print("  AMT_INCOME_TOTAL not found, added random dummy LOAN_REPAYMENT_SCORE.")

    print("\n--- Scored Data Schema ---")
    scored_data.printSchema()
    return scored_data


def define_scoring_expectations(
    context, suite_name=LOAN_DEFAULT_PREDICTIONS_SUITE_NAME
):
    """Defines Great Expectations suite for the loan default scoring model results."""
    suite = ExpectationSuite(expectation_suite_name=suite_name)

    # Expectations for the final score:
    # 1. Ensure the final score is never null.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(  # Reverted to short name via gx.expectations
            column="LOAN_REPAYMENT_SCORE",
            meta={
                "severity": CRITICAL_SEVERITY,
                "description": "Ensure the final score is never null.",
            },
        )
    )
    # 2. Ensure the final score is between 0 and 100.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(  # Reverted to short name via gx.expectations
            column="LOAN_REPAYMENT_SCORE",
            min_value=0,
            max_value=100,
            meta={
                "severity": CRITICAL_SEVERITY,
                "description": "Ensure the final score is between 0 and 100.",
            },
        )
    )
    # 3. Ensure the final score is numeric.
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeOfType(  # Reverted to short name via gx.expectations
            column="LOAN_REPAYMENT_SCORE",
            type="float",  # Consistent with previous numeric type checks
            meta={
                "severity": CRITICAL_SEVERITY,
                "description": "Ensure the final score is numeric.",
            },
        )
    )
    context.add_expectation_suite(suite)
    context.save_expectation_suite(suite)
    print(f"Defined and saved ExpectationSuite: {suite_name}")
    return suite


def main():
    """Main function to execute the loan default prediction pipeline."""
    # Initialize Spark session
    spark = create_spark_session()
    print("Spark Session initialized.")

    # Initialize Great Expectations context
    context = gx.get_context()
    print("Great Expectations Data Context initialized.")

    # Read source data
    try:
        application_train_df = read_data(spark, APPLICATION_TRAIN_PATH)
        bureau_df = read_data(spark, BUREAU_PATH)
    except IOError as e:
        print(f"Fatal Error: Failed to read input data: {e}")
        spark.stop()
        return

    # --- JOIN Transformation ---
    print("\n--- Step: Joining DataFrames ---")
    joined_df = perform_join(application_train_df, bureau_df)
    # Persist the DataFrame in memory (cache) before multiple operations/validations
    joined_df.cache()
    # Count to trigger caching
    print(f"Joined DataFrame has {joined_df.count()} rows.")

    # --- Data Quality Checks for JOIN ---
    print("\n--- Step: Validating Joined Data ---")
    join_suite = define_join_expectations(context)
    if not validate_data(context, joined_df, join_suite.expectation_suite_name):
        print("Joint data validation failed. Exiting.")
        spark.stop()
        return

    # --- Column Manipulation: Impute Missing Values ---
    print("\n--- Step: Imputing Missing Values ---")
    cleaned_data = impute_missing_values(joined_df)
    cleaned_data.cache()
    print(f"Cleaned Data has {cleaned_data.count()} rows.")

    # --- Data Quality Checks for Column Manipulation (Imputation) ---
    print("\n--- Step: Validating Cleaned Data ---")
    imputation_suite = define_imputation_expectations(context)
    if not validate_data(
        context, cleaned_data, imputation_suite.expectation_suite_name
    ):
        print("Cleaned data validation failed. Exiting.")
        spark.stop()
        return

    # --- Column Manipulation: Apply Loan Default Scoring Model ---
    print("\n--- Step: Applying Loan Default Scoring Model ---")
    scored_data = apply_loan_scoring_model(cleaned_data)
    scored_data.cache()
    print(f"Scored Data has {scored_data.count()} rows.")

    # --- Output: Loan Default Scoring Model Results ---
    loan_default_predictions = scored_data  # Renaming for clarity as final output

    # --- Data Quality Checks for Output ---
    print("\n--- Step: Validating Final Output Data ---")
    scoring_suite = define_scoring_expectations(context)
    if not validate_data(
        context, loan_default_predictions, scoring_suite.expectation_suite_name
    ):
        print("Final output data validation failed. Exiting.")
        spark.stop()
        return

    # Final step: Build and open Great Expectations Data Docs
    print("\n--- Building Great Expectations Data Docs ---")
    # This ensures docs are generated at the end, even if interim validations succeed
    context.build_data_docs()
    # You can open them by running `great_expectations docs open` in your GE project directory.

    # Write output data to Parquet format
    print(f"\n--- Writing Final Output to {OUTPUT_PATH} ---")
    try:
        loan_default_predictions.write.mode("overwrite").parquet(OUTPUT_PATH)
        print(f"Successfully wrote final output to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Fatal Error: Failed to write data to {OUTPUT_PATH}: {e}")
    finally:
        # Stop the Spark session cleanly
        print("Stopping Spark Session.")
        spark.stop()


if __name__ == "__main__":
    main()
