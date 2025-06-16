from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, mean
from pyspark.sql.types import IntegerType, DoubleType, FloatType, StringType

# Initialize Spark session
spark = SparkSession.builder.appName("LoanDefaultPrediction").getOrCreate()

# Define file paths
APPLICATION_TRAIN_PATH = "data/raw/application_train.csv"
BUREAU_PATH = "data/raw/bureau.csv"
OUTPUT_PATH = "data/output/loan_default_predictions"

def read_data(spark, path):
    """Reads a CSV file into a Spark DataFrame.

    Args:
        spark: SparkSession.
        path: Path to the CSV file.

    Returns:
        Spark DataFrame.  Returns None and logs an error if the read fails.
    """
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def transform_data(application_train, bureau):
    """Transforms the input DataFrames.

    Args:
        application_train: application_train DataFrame.
        bureau: bureau DataFrame.

    Returns:
        Transformed DataFrame.
    """
    # Join application_train with bureau on SK_ID_CURR
    joined_data = application_train.join(bureau, on="SK_ID_CURR", how="inner")

    # Handle missing values:  Calculate means only once
    numerical_cols = [field.name for field in joined_data.schema.fields if field.dataType in [IntegerType, DoubleType, FloatType]]
    means = {}
    for col_name in numerical_cols:
        means[col_name] = joined_data.select(mean(col(col_name))).first()[0]

    joined_data = joined_data.fillna(means)

    # For string columns, fill missing values with 'Unknown'
    string_cols = [field.name for field in joined_data.schema.fields if field.dataType == StringType]
    joined_data = joined_data.fillna({'Unknown': 'Unknown'})
    
    # Create risk score
    joined_data = joined_data.withColumn(
        "RISK_SCORE",
        when((col("AMT_INCOME_TOTAL") / col("AMT_CREDIT") > 0.5) & (col("CNT_CHILDREN") < 2), 0.2)
        .when((col("AMT_INCOME_TOTAL") / col("AMT_CREDIT") > 0.25) & (col("CNT_CHILDREN") < 3), 0.5)
        .otherwise(0.8)
    )
    
    return joined_data


def write_data(df, path):
    """Writes a Spark DataFrame to a CSV file.

    Args:
        df: Spark DataFrame to write.
        path: Output path.
    """
    try:
        df.write.csv(path, header=True, mode="overwrite")
    except Exception as e:
        print(f"Error writing to {path}: {e}")

def main():
    """Main function to execute the loan default prediction pipeline."""
    # Read source data
    application_train = read_data(spark, APPLICATION_TRAIN_PATH)
    bureau = read_data(spark, BUREAU_PATH)

    # Check if DataFrames were loaded successfully
    if application_train is None or bureau is None:
        print("Failed to load data. Exiting.")
        spark.stop()
        return

    # Data Transformation
    transformed_data = transform_data(application_train, bureau)

    # Output
    transformed_data.select("SK_ID_CURR", "RISK_SCORE").show()
    write_data(transformed_data, OUTPUT_PATH)

    spark.stop()

if __name__ == "__main__":
    main()