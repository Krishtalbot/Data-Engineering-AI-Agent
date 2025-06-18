import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, lit, mode
from pyspark.sql.types import DoubleType
import great_expectations as gx
from great_expectations.dataset import SparkDFDataset


def read_data(spark, file_path, file_name):
    """Reads a CSV file into a Spark DataFrame.

    Args:
        spark (SparkSession): The SparkSession.
        file_path (str): The path to the CSV file.
        file_name (str): name of the csv file for logging purposes

    Returns:
        DataFrame: The Spark DataFrame.
    """
    try:
        df = spark.read.csv(file_path, header=True, inferSchema=True)
        print(f"Successfully read data from {file_name}: {file_path}")
        return df
    except Exception as e:
        print(f"Error reading {file_name} file: {e}")
        sys.exit(1)


def create_great_expectations_suite(context, suite_name, expectations):
    """Creates a Great Expectations suite and adds the specified expectations.

    Args:
        context (DataContext): The Great Expectations DataContext.
        suite_name (str): The name of the expectation suite.
        expectations (list): A list of Great Expectations expectation objects.
    """
    try:
        suite = context.create_expectation_suite(expectation_suite_name=suite_name, overwrite_existing=True)
        for expectation in expectations:
            suite.add_expectation(expectation)
        context.save_expectation_suite(suite)
        print(f"Successfully created and saved Great Expectations suite: {suite_name}")
    except Exception as e:
        print(f"Error creating Great Expectations suite {suite_name}: {e}")
        sys.exit(1)


def validate_data(context, df, suite_name, df_name):
    """Validates a Spark DataFrame against a Great Expectations suite.

    Args:
        context (DataContext): The Great Expectations DataContext.
        df (DataFrame): The Spark DataFrame to validate.
        suite_name (str): The name of the expectation suite.
        df_name (str): The name of the dataframe for logging purposes.
    """
    try:
        ge_df = SparkDFDataset(df)
        validation_results = ge_df.validate(expectation_suite_name=suite_name)

        if not validation_results["success"]:
            print(f"Data quality checks failed for {df_name}!")
            for result in validation_results["results"]:
                if not result["success"]:
                    print(f"  Expectation: {result['expectation_config']['expectation_type']}")
                    print(f"  Column: {result['expectation_config']['kwargs'].get('column', 'N/A')}")
                    print(f"  Result: {result['result']}")
                    if result['expectation_config'].get('severity', 'WARNING') == "CRITICAL":
                        raise Exception(f"Critical data quality check failed for {df_name}.")

        context.build_data_docs()
        print(f"Successfully validated data against Great Expectations suite: {suite_name}")

    except Exception as e:
        print(f"Error during data quality checks for {df_name}: {e}")
        sys.exit(1)


def handle_missing_values(df):
    """Handles missing values in a Spark DataFrame by imputing with mean (numeric) or mode (string).

    Args:
        df (DataFrame): The input Spark DataFrame.

    Returns:
        DataFrame: The DataFrame with missing values imputed.
    """
    cleaned_data = df
    # Impute missing numeric values with column mean.
    for col_name in cleaned_data.columns:
        if cleaned_data.schema[cleaned_data.columns.index(col_name)].dataType == DoubleType():  # check if column is of DoubleType
            mean_value = cleaned_data.agg(mean(col(col_name))).collect()[0][0]
            if mean_value is not None:  # handles the case where the whole column is null
                cleaned_data = cleaned_data.fillna(mean_value, subset=[col_name])

    # Impute missing categorical values with column mode.
    for col_name in cleaned_data.columns:
        if cleaned_data.schema[cleaned_data.columns.index(col_name)].dataType.typeName() == 'string':  # check if column is of StringType
            mode_value = cleaned_data.groupBy(col_name).count().orderBy(col("count"), ascending=False).first()[col_name]
            if mode_value is not None:  # handles the case where the whole column is null
                cleaned_data = cleaned_data.fillna(mode_value, subset=[col_name])

    return cleaned_data


def main():
    """Main ETL function for loan scoring."""
    # Initialize Spark session
    spark = SparkSession.builder.appName("LoanScoringETL").getOrCreate()

    # Source data filenames
    application_train_file = "data/raw/application_train.csv"
    bureau_file = "data/raw/bureau.csv"

    # Output file
    output_file = "data/output/loan_scores.parquet"

    # Read source data
    application_train_df = read_data(spark, application_train_file, "application_train")
    bureau_df = read_data(spark, bureau_file, "bureau")

    # Initialize Great Expectations context
    context = gx.get_context()

    # Transformation: Join application_train and bureau tables
    join_key = "SK_ID_CURR"
    join_type = "inner"  # or "left", "right", etc.
    joined_df = application_train_df.join(bureau_df, on=join_key, how=join_type)
    joined_df.cache()  # Cache the joined DataFrame as it will be used for validation and further processing

    # Data Quality Checks - Post Join
    joined_df_suite_name = "joined_df_suite"
    create_great_expectations_suite(
        context,
        joined_df_suite_name,
        [
            gx.expectations.ExpectColumnValuesToBePresent(column=join_key),
            gx.expectations.ExpectColumnValuesToBeUnique(
                column=join_key, ignore_row_if="any_value_is_missing"
            ),
        ],
    )
    validate_data(context, joined_df, joined_df_suite_name, "joined_df")

    # Transformation: Handle missing values
    cleaned_data = handle_missing_values(joined_df)
    cleaned_data.cache()  # Cache the cleaned DataFrame

    # Data Quality Checks - Post Cleaning
    cleaned_data_suite_name = "cleaned_data_suite"
    create_great_expectations_suite(
        context,
        cleaned_data_suite_name,
        [
            gx.expectations.ExpectColumnValuesToBePresent(column="AMT_INCOME_TOTAL"),
            gx.expectations.ExpectColumnValuesToBeOfType(column="AMT_INCOME_TOTAL", type_="numeric"),
        ],
    )
    validate_data(context, cleaned_data, cleaned_data_suite_name, "cleaned_data")

    # Transformation: Apply loan default scoring model
    scored_data = cleaned_data  # placeholder
    # Assume 'your_automl_model_function' takes a Spark DataFrame and returns a DataFrame with a 'LOAN_REPAYMENT_SCORE' column.
    # scored_data = your_automl_model_function(cleaned_data)

    # Placeholder for adding LOAN_REPAYMENT_SCORE
    scored_data = scored_data.withColumn("LOAN_REPAYMENT_SCORE", lit(50).cast("integer"))
    scored_data.cache()

    # Output: Apply scoring model and generate final output
    final_output = scored_data
    final_output_suite_name = "final_output_suite"

    create_great_expectations_suite(
        context,
        final_output_suite_name,
        [
            gx.expectations.ExpectColumnValuesToBePresent(column="LOAN_REPAYMENT_SCORE"),
            gx.expectations.ExpectColumnValuesToBeBetween(column="LOAN_REPAYMENT_SCORE", min_value=0, max_value=100),
            gx.expectations.ExpectColumnValuesToBeOfType(column="LOAN_REPAYMENT_SCORE", type_="numeric"),
        ],
    )

    validate_data(context, final_output, final_output_suite_name, "final_output")

    # Global Data Quality Checks
    global_suite_name = "global_suite"
    create_great_expectations_suite(
        context,
        global_suite_name,
        [gx.expectations.ExpectTableRowCountToBeBetween(min_value=1000, max_value=400000)],
    )

    validate_data(context, final_output, global_suite_name, "final_output_global")

    # Write output data
    final_output.write.parquet(output_file, mode="overwrite")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()