from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder.appName("LoanDefaultPrediction").getOrCreate()

def read_data(spark, path):
    """Reads a CSV file into a Spark DataFrame with schema inference."""
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        print(f"Successfully read data from {path}")
        return df
    except Exception as e:
        print(f"Error reading data from {path}: {e}")
        return None

def inspect_dataframe(df, name):
    """Prints the schema and row count of a DataFrame."""
    print(f"{name} schema:")
    df.printSchema()
    print(f"{name} row count: {df.count()}")

def handle_missing_values(df):
    """Fills missing values in a DataFrame.
    Numeric columns are filled with their mean, and string columns with 'Unknown'.
    """
    for column in df.columns:
        col_type = df.schema[column].dataType
        if isinstance(col_type, StringType):
            df = df.fillna({column: 'Unknown'}, subset=[column])
        else:
            mean_val = df.select(column).agg({column: 'mean'}).first()[0]
            fill_value = mean_val if mean_val is not None else 0  # Use 0 if mean is None
            df = df.fillna({column: fill_value}, subset=[column])
    return df

def predict_loan_default(df):
    """Predicts loan default based on income and credit history."""
    return df.withColumn(
        "predicted_default",
        when((col("AMT_INCOME_TOTAL") > 100000) & (col("CNT_CREDIT_PROLONG") == 0), 0).otherwise(1)
    )

def write_output(df, path):
    """Writes the DataFrame to a CSV file."""
    try:
        df.write.csv(path, header=True, mode="overwrite")
        print(f"Successfully wrote data to {path}")
    except Exception as e:
        print(f"Error writing data to {path}: {e}")

# Main execution
def main():
    # Load data
    application_train = read_data(spark, "data/raw/application_train.csv")
    bureau = read_data(spark, "data/raw/bureau.csv")

    # Check if DataFrames were loaded successfully
    if application_train is None or bureau is None:
        print("One or more dataframes failed to load. Exiting.")
        return

    # Inspect DataFrames
    inspect_dataframe(application_train, "application_train")
    inspect_dataframe(bureau, "bureau")

    # Join DataFrames
    joined_df = application_train.join(bureau, "SK_ID_CURR", "inner")

    # Handle missing values
    joined_df = handle_missing_values(joined_df)

    # Predict loan default
    joined_df = predict_loan_default(joined_df)

    # Display results
    joined_df.show()

    # Write output
    write_output(joined_df, "data/output/loan_default_predictions.csv")

if __name__ == "__main__":
    main()

    # Stop Spark session
    spark.stop()