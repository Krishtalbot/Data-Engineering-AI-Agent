from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, lit
from pyspark.sql.types import StringType
import os

# Initialize Spark session
spark = SparkSession.builder.appName("LoanDefaultPrediction").getOrCreate()

# Define source file paths
application_train_path = "data/raw/application_train.csv"
bureau_path = "data/raw/bureau.csv"


def read_data(spark, path):
    """Reads a CSV file into a Spark DataFrame with error handling."""
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return None


# Read source data
application_train = read_data(spark, application_train_path)
bureau = read_data(spark, bureau_path)

# Check if DataFrames are valid before proceeding
if application_train is None or bureau is None:
    spark.stop()
    exit()

# Print schemas for verification
print("Application Train Schema:")
application_train.printSchema()
print("Bureau Schema:")
bureau.printSchema()

# Join application_train with bureau on SK_ID_CURR using inner join
joined_data = application_train.join(bureau, "SK_ID_CURR", "inner")

# Cache the joined data for faster subsequent operations
joined_data.cache()


def handle_missing_values(df):
    """Replaces null values in a DataFrame with 0 for numeric columns and 'unknown' for string columns."""
    for column in df.columns:
        if isinstance(df.schema[column].dataType, StringType):
            df = df.withColumn(column, when(col(column).isNull(), lit('unknown')).otherwise(col(column)))
        else:
            df = df.withColumn(column, when(col(column).isNull(), lit(0)).otherwise(col(column)))
    return df


# Handle missing values
joined_data = handle_missing_values(joined_data)

# Verify no nulls (optimized)
def verify_no_nulls(df):
    """Verifies that there are no null values in the DataFrame."""
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        print(f"Null count for {column}: {null_count}")
        if null_count > 0:
            print(f"Warning: Null values found in column {column}")

verify_no_nulls(joined_data)


# --- Loan Default Prediction Model ---
# In a real-world scenario, this would involve loading a pre-trained model
# and applying it to the joined data. Since a pre-trained model and the
# specifics of the model were not provided, this placeholder generates a
# simple prediction based on arbitrary conditions.

# Example: Predict loan default based on some conditions
# NOTE: This is a simplified example and should be replaced with a real scoring model.
joined_data = joined_data.withColumn(
    "predicted_default",
    when((col("AMT_CREDIT") > 500000) & (col("AMT_INCOME_TOTAL") < 50000), lit(1)).otherwise(lit(0))
)

# The 'predicted_default' column now contains a binary prediction (1 for default, 0 for no default).
# --- End of Loan Default Prediction Model ---

# Output results as a table (display the first 20 rows)
joined_data.show()

# (Optional) Write the output to a file
output_path = "data/output/loan_default_predictions.csv"

# Create the directory if it doesn't exist
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    joined_data.coalesce(1).write.csv(output_path, header=True, mode="overwrite")  # coalesce to single partition
except Exception as e:
    print(f"Error writing to file {output_path}: {e}")

# Stop Spark session
spark.stop()