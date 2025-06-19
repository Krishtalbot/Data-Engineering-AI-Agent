import pandas as pd
import os
from pinecone_conf import PineconeManager, DataInput
from pinecone_index import ensure_pinecone_index
from dotenv import load_dotenv
import logging

# Import the new Spark utility
from schema_utils import get_dataframe_schema_spark, get_spark_session

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Define the base directory for your actual data files
DATA_FILES_BASE_DIR = "data/raw"

# Cache for schemas to avoid re-reading large files repeatedly
_schema_cache_spark = {}


def process_csv_and_store(csv_file_path: str, index_manager: PineconeManager):
    spark = get_spark_session()  # This initializes it if it hasn't been already

    if not os.path.exists(csv_file_path):
        logging.error(
            f"Error: CSV description file not found at {csv_file_path}. Please ensure it's in the correct directory."
        )
        return

    try:
        df_desc = pd.read_csv(csv_file_path, encoding="latin1")
        logging.info(
            f"Successfully loaded description CSV file: {csv_file_path} with encoding 'latin1'."
        )
        logging.info(f"Description CSV columns found: {df_desc.columns.tolist()}")

        required_columns = ["Table", "Row", "Description"]
        if not all(col in df_desc.columns for col in required_columns):
            logging.error(
                f"Missing one or more required columns ({required_columns}) in description CSV. Found: {df_desc.columns.tolist()}"
            )
            return

        logging.info(
            f"Starting to process {len(df_desc)} rows and store in Pinecone index '{index_manager.index_name}'..."
        )

        for idx, row in df_desc.iterrows():
            table_name_raw = str(row["Table"])
            column_name = str(row["Row"])
            description = str(row["Description"])

            actual_table_names = []
            if "{train|test}" in table_name_raw:
                actual_table_names.append(
                    table_name_raw.replace("{train|test}", "train")
                )
                actual_table_names.append(
                    table_name_raw.replace("{train|test}", "test")
                )
            else:
                actual_table_names.append(table_name_raw)

            for current_table_name_base in actual_table_names:
                if not current_table_name_base.endswith(".csv"):
                    current_table_name_base += ".csv"

                full_data_file_path = os.path.join(
                    DATA_FILES_BASE_DIR, current_table_name_base
                )

                # Get schema from cache or infer it using Spark
                if full_data_file_path not in _schema_cache_spark:
                    _schema_cache_spark[full_data_file_path] = (
                        get_dataframe_schema_spark(full_data_file_path)
                    )

                column_schema = _schema_cache_spark.get(full_data_file_path, {}).get(
                    column_name, {}
                )
                data_type = column_schema.get("data_type", "Unknown")
                nullable = column_schema.get("nullable", False)

                text_to_embed = (
                    f"Table: {current_table_name_base}, Column: {column_name}, "
                    f"Description: {description}, Data Type: {data_type}, Nullable: {nullable}"
                )

                metadata = {
                    "table": current_table_name_base,
                    "column": column_name,
                    "original_description": description,
                    "data_type": data_type,
                    "nullable": nullable,
                }

                data_input = DataInput(text=text_to_embed, metadata=metadata)
                response = index_manager.store_data(data_input)

                if response:
                    logging.info(
                        f"Row {idx + 1}: Stored '{column_name}' from table '{current_table_name_base}' "
                        f"(Inferred Type: {data_type}, Nullable: {nullable}). Chunks: {response.get('chunks')}"
                    )
                else:
                    logging.warning(
                        f"Row {idx + 1}: Failed to store '{column_name}' from table '{current_table_name_base}'."
                    )

        logging.info("Finished processing and storing all data from description CSV.")

    except pd.errors.EmptyDataError:
        logging.error(f"The CSV description file '{csv_file_path}' is empty.")
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing description CSV file '{csv_file_path}': {e}")
    except UnicodeDecodeError as e:
        logging.error(
            f"UnicodeDecodeError: Failed to decode description CSV file with 'latin1' encoding. "
            f"It might be encoded differently. Original error: {e}"
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during CSV processing: {str(e)}")
    finally:
        # Stop SparkSession at the very end of your script or when no longer needed
        # spark.stop() # Only uncomment this if you are absolutely sure no other parts of your app need Spark
        pass


if __name__ == "__main__":
    csv_file_path = "data/raw/HomeCredit_columns_description.csv"
    # Ensure your actual data files (e.g., application_train.csv) are in 'data/raw'
    # For testing, you might need to create dummy files as shown in Pandas example.

    new_index_name = input(
        "Enter the name of the Pinecone index you want to use (e.g., 'home-credit-columns-data'): "
    ).strip()
    if not new_index_name:
        print("Index name cannot be empty. Exiting.")
        exit(1)

    print(f"\nEnsuring Pinecone index '{new_index_name}' exists or creating it...")
    try:
        ensure_pinecone_index(new_index_name)
    except Exception as e:
        print(
            f"Failed to ensure Pinecone index exists: {e}. Please check your Pinecone API key and network connection."
        )
        exit(1)

    print(f"\nInitializing PineconeManager for index: '{new_index_name}'...")
    try:
        manager = PineconeManager(new_index_name)
    except Exception as e:
        print(f"Error initializing PineconeManager: {e}.")
        print(
            "Please ensure your PINECONE_API_KEY is set in your .env file and the 'all-mpnet-base-v2' model is downloaded correctly."
        )
        exit(1)

    print(
        f"\nStarting to process '{csv_file_path}' and store data in Pinecone index '{new_index_name}'..."
    )
    process_csv_and_store(csv_file_path, manager)

    print(f"\nData storage process complete for index '{new_index_name}'.")

    print("\n--- Test Search Functionality ---")
    while True:
        query = input(
            f"Enter a query to search for similar column descriptions in '{new_index_name}' (type 'exit' to quit): "
        ).strip()
        if query.lower() == "exit":
            # Stop Spark Session gracefully when exiting the script
            get_spark_session().stop()
            break
        if not query:
            print("Query cannot be empty. Please enter something to search.")
            continue

        print(f"Searching for: '{query}' in index '{new_index_name}'...")
        try:
            results = manager.search_similar_prompt(query)
            if results and results.matches:
                print("\nFound similar descriptions:")
                for match in results.matches:
                    score = match.score
                    original_desc = match.metadata.get("original_description", "N/A")
                    table = match.metadata.get("table", "N/A")
                    column = match.metadata.get("column", "N/A")
                    data_type = match.metadata.get("data_type", "N/A")
                    nullable = match.metadata.get("nullable", "N/A")
                    print(
                        f"- Score: {score:.4f} | Table: {table} | Column: {column} | Data Type: {data_type} | Nullable: {nullable} | Description: {original_desc}"
                    )
            else:
                print(
                    "No similar descriptions found or an error occurred during search."
                )
        except Exception as e:
            print(f"An error occurred during search: {e}")
