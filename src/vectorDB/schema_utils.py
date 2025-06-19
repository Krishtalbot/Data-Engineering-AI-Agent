# schema_utils_spark.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull
import os
import logging

logging.basicConfig(level=logging.INFO)


# Initialize Spark Session (ensure only one is active)
def get_spark_session():
    if "spark" not in globals():
        globals()["spark"] = (
            SparkSession.builder.appName("SchemaInference")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .getOrCreate()
        )
    return globals()["spark"]


def get_dataframe_schema_spark(file_path: str) -> dict:
    """
    Infers data types and nullability for columns in a given CSV file using PySpark.

    Args:
        file_path (str): The path to the data CSV file.

    Returns:
        dict: A dictionary where keys are column names and values are
              another dictionary containing 'data_type' and 'nullable' status.
              Returns an empty dict if the file cannot be processed.
    """
    spark = get_spark_session()

    if not os.path.exists(file_path):
        logging.warning(f"Data file not found at {file_path}. Cannot infer schema.")
        return {}

    try:
        # For Spark, it's common to let it infer schema for CSVs.
        # inferSchema=True makes Spark read the file once to determine types.
        # header=True assumes the first row is a header.
        df_spark = spark.read.csv(file_path, header=True, inferSchema=True)
        logging.info(
            f"Successfully loaded {file_path} with schema inference using Spark."
        )
        df_spark.printSchema()  # Good for debugging to see inferred schema

        schema_info = {}
        for spark_field in df_spark.schema:
            col_name = spark_field.name
            data_type = str(
                spark_field.dataType
            )  # Spark data type (e.g., IntegerType, StringType, DoubleType)

            # To check nullability, we need to run a count on nulls.
            # This can be expensive for very wide tables, but generally efficient.
            null_count = df_spark.filter(col(col_name).isNull()).count()
            nullable = bool(null_count > 0)

            schema_info[col_name] = {"data_type": data_type, "nullable": nullable}
        return schema_info
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during schema inference for {file_path} with Spark: {str(e)}"
        )
        return {}
