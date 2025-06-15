from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.types import IntegerType, DoubleType, FloatType

# Initialize Spark session
spark = SparkSession.builder.appName("LoanDefaultPrediction").getOrCreate()

# Define file paths
APPLICATION_TRAIN_PATH = "data/raw/application_train.csv"
BUREAU_PATH = "data/raw/bureau.csv"

def read_data(spark, path):
    """
    Reads a CSV file into a Spark DataFrame.

    Args:
        spark: SparkSession.
        path (str): Path to the CSV file.

    Returns:
        DataFrame: Spark DataFrame.
    """
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        print(f"{path} read successfully.")
        return df
    except Exception as e:
        print(f"Error reading {path}: {e}")
        raise  # Re-raise the exception to halt execution

# Read data
application_train = read_data(spark, APPLICATION_TRAIN_PATH)
bureau = read_data(spark, BUREAU_PATH)

# Join DataFrames
try:
    joined_data = application_train.join(bureau, on="SK_ID_CURR", how="inner")
    print("DataFrames joined successfully.")
except Exception as e:
    print(f"Error joining dataframes: {e}")
    raise

# Identify numerical and string columns
NUMERIC_TYPES = [IntegerType, DoubleType, FloatType]
numerical_cols = [field.name for field in joined_data.schema.fields if any(isinstance(field.dataType, t) for t in NUMERIC_TYPES)]
string_cols = [field.name for field in joined_data.schema.fields if field.dataType.typeName() == 'string']

def impute_missing_values(df, numerical_cols, string_cols):
    """
    Imputes missing values in a DataFrame.

    Args:
        df: Spark DataFrame.
        numerical_cols (list): List of numerical column names.
        string_cols (list): List of string column names.

    Returns:
        DataFrame: DataFrame with imputed values.
    """
    # Impute numerical columns with the mean
    for col_name in numerical_cols:
        mean_val = df.select(mean(col(col_name))).first()[0]
        df = df.fillna({col_name: mean_val})

    # Impute string columns with "Unknown"
    for col_name in string_cols:
        df = df.fillna({col_name: "Unknown"})
    return df

# Impute missing values
joined_data = impute_missing_values(joined_data, numerical_cols, string_cols)
print("Missing values imputed successfully.")

# Define feature and target columns
FEATURE_COLS = [col_name for col_name in joined_data.columns if col_name not in ["SK_ID_CURR", "TARGET"]]
TARGET_COL = "TARGET"

# Assemble features
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
assembled_data = assembler.transform(joined_data).cache() # Cache here

# Train the model
lr = LogisticRegression(featuresCol="features", labelCol=TARGET_COL)
pipeline = Pipeline(stages=[lr])

try:
    model = pipeline.fit(assembled_data)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error training model: {e}")
    raise

# Apply scoring model
predictions = model.transform(assembled_data)

# Select output columns
output_table = predictions.select("SK_ID_CURR", "prediction", "probability")

# Show results
output_table.show()

# Optional: Save output (consider using a more robust format like Parquet for large datasets)
# output_table.write.parquet("data/output/loan_default_predictions.parquet")

print("Loan default predictions completed.")

spark.stop()