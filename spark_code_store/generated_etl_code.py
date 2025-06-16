from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, lit, mean, desc, first
from pyspark.sql.types import IntegerType, DoubleType, FloatType, StringType


# Initialize Spark session
spark = SparkSession.builder.appName("LoanDefaultPrediction").getOrCreate()

# Define file paths
APPLICATION_TRAIN_PATH = "data/raw/application_train.csv"
BUREAU_PATH = "data/raw/bureau.csv"
OUTPUT_PATH = "data/output/loan_default_predictions"


def read_data(spark, path):
    """Reads a CSV file into a Spark DataFrame."""
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None


def handle_missing_values(df):
    """Imputes missing numerical values with the mean and categorical values with the most frequent value."""

    # Define numerical and categorical type names
    numerical_types = [IntegerType, DoubleType, FloatType]
    string_type = StringType

    # Identify numerical and categorical columns
    numerical_cols = [
        field.name for field in df.schema.fields if field.dataType in numerical_types
    ]
    categorical_cols = [
        field.name for field in df.schema.fields if field.dataType == string_type
    ]

    # Impute numerical columns with the mean
    for col_name in numerical_cols:
        mean_value = df.select(mean(col(col_name))).first()[0]
        df = df.fillna({col_name: mean_value})

    # Impute categorical columns with the most frequent value
    for col_name in categorical_cols:
        most_frequent_value = (
            df.groupBy(col_name)
            .count()
            .orderBy(desc("count"))
            .select(first(col_name))
            .first()[0]
        )
        df = df.fillna({col_name: most_frequent_value})
    return df


def predict_loan_default(df):
    """Generates a dummy prediction column based on EXT_SOURCE_3."""
    # Dummy prediction: predict based on EXT_SOURCE_3, replace with real model prediction
    return df.withColumn(
        "predicted_default", when(col("EXT_SOURCE_3") > 0.5, lit(1)).otherwise(lit(0))
    )


def write_output(df, path):
    """Writes the DataFrame to a CSV file."""
    try:
        df.write.csv(path, header=True, mode="overwrite")
        print(f"Output data written to {path}")
    except Exception as e:
        print(f"Error writing output to {path}: {e}")


def main():
    """Main function to orchestrate the loan default prediction process."""
    # Read source data
    application_train = read_data(spark, APPLICATION_TRAIN_PATH)
    bureau = read_data(spark, BUREAU_PATH)

    if not application_train or not bureau:
        print("Failed to load data. Exiting.")
        spark.stop()
        return

    # Join application_train with bureau
    joined_data = application_train.join(bureau, "SK_ID_CURR", "inner")

    # Handle missing values
    joined_data = handle_missing_values(joined_data)

    # Business Rules: Predict loan default
    joined_data = predict_loan_default(joined_data)

    # Output: Apply scoring model and output results as a table
    output_data = joined_data.select("SK_ID_CURR", "predicted_default")

    # Write the output data to a file
    write_output(output_data, OUTPUT_PATH)


if __name__ == "__main__":
    main()
    spark.stop()