```python
# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, lit, mean
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml import PipelineModel

# Set up a Spark session
spark = SparkSession.builder.appName("LoanDefaultPrediction").getOrCreate()

# Define file paths for the source data
application_train_path = "data/raw/application_train.csv"
bureau_path = "data/raw/bureau.csv"

# Read the source data files
try:
    application_train = spark.read.csv(application_train_path, header=True, inferSchema=True)
    bureau = spark.read.csv(bureau_path, header=True, inferSchema=True)
except Exception as e:
    print(f"Error reading source data: {e}")
    spark.stop()
    exit()

# Print schema and number of rows of the dataframes
application_train.printSchema()
print("application_train count: ", application_train.count())
bureau.printSchema()
print("bureau count: ", bureau.count())

# Transformation: Join application_train with bureau on SK_ID_CURR using inner join
try:
    combined_data = application_train.join(bureau, on="SK_ID_CURR", how="inner")
    print("combined_data count: ", combined_data.count())
except Exception as e:
    print(f"Error joining dataframes: {e}")
    spark.stop()
    exit()

# Transformation: Handle missing values using imputation
# Impute missing numerical values with the mean of the column
numerical_cols = [field.name for field in combined_data.schema.fields if field.dataType.typeName() in ['integer', 'double', 'float']]

for col_name in numerical_cols:
    mean_val = combined_data.select(mean(col_name)).first()[0]
    combined_data = combined_data.fillna({col_name: mean_val})

# Impute missing string values with "Unknown"
string_cols = [field.name for field in combined_data.schema.fields if field.dataType.typeName() == 'string']

for col_name in string_cols:
    combined_data = combined_data.fillna({col_name: "Unknown"})

# Business Rules: Predict loan default based on applicant data and credit history
# Load the pre-trained scoring model.  Assumes model exists in specified path.
model_path = "model/loan_default_model" # Replace with the actual path to your model
try:
    loaded_model = PipelineModel.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    spark.stop()
    exit()

# Apply the model to the transformed data
try:
    predictions = loaded_model.transform(combined_data)
except Exception as e:
    print(f"Error applying model: {e}")
    spark.stop()
    exit()

# Output: Apply scoring model for loan default prediction
# Select relevant columns for the output
output_df = predictions.select("SK_ID_CURR", "prediction", "probability")

# Output: Output results as a table
output_path = "data/output/loan_default_predictions"  # Replace with your desired output path

# Write the output data to a CSV file
try:
    output_df.write.csv(output_path, header=True, mode="overwrite") # Overwrite to prevent appending
    print(f"Output data written to: {output_path}")

except Exception as e:
    print(f"Error writing output data: {e}")
    spark.stop()
    exit()

# Stop the Spark session
spark.stop()
```

FINISH
