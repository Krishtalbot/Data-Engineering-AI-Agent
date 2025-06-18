import pandas as pd
import os
from pinecone_conf import PineconeManager, DataInput  # Import the new PineconeManager
from pinecone_index import ensure_pinecone_index  # Import the index creation utility
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)


def process_csv_and_store(csv_file_path: str, index_manager: PineconeManager):
    """
    Reads a CSV file containing column descriptions, formats the data,
    and stores it in the Pinecone index managed by the provided index_manager.

    Args:
        csv_file_path (str): The path to the CSV file.
        index_manager (PineconeManager): An instance of PineconeManager
                                         configured for the target index.
    """
    if not os.path.exists(csv_file_path):
        logging.error(
            f"Error: CSV file not found at {csv_file_path}. Please ensure it's in the correct directory."
        )
        return

    try:
        df = pd.read_csv(csv_file_path, encoding="latin1")
        logging.info(
            f"Successfully loaded CSV file: {csv_file_path} with encoding 'latin1'."
        )
        logging.info(f"CSV columns found: {df.columns.tolist()}")

        required_columns = ["Table", "Row", "Description"]
        if not all(col in df.columns for col in required_columns):
            logging.error(
                f"Missing one or more required columns ({required_columns}) in CSV. Found: {df.columns.tolist()}"
            )
            return

        logging.info(
            f"Starting to process {len(df)} rows and store in Pinecone index '{index_manager.index_name}'..."
        )
        for idx, row in df.iterrows():
            table_name_raw = str(row["Table"])
            column_name = str(row["Row"])
            description = str(row["Description"])

            # --- MODIFICATION START ---
            # Handle the dynamic table name "application_{train|test}.csv"
            # If the raw table name contains "{train|test}", create entries for both train and test.
            if "{train|test}" in table_name_raw:
                # Create a list of actual table names to process
                actual_table_names = [
                    table_name_raw.replace("{train|test}", "train"),
                    table_name_raw.replace("{train|test}", "test"),
                ]
            else:
                # If no placeholder, just use the table_name_raw as is
                actual_table_names = [table_name_raw]

            # Iterate over the determined actual table names
            for current_table_name in actual_table_names:
                # Append '.csv' to the table name if it's not already there, for consistency
                # Assuming your raw data files are named like 'application_train.csv'
                if not current_table_name.endswith(".csv"):
                    current_table_name += ".csv"

                # Combine relevant information into a single text string for embedding
                # This text will be embedded and used for semantic search
                text_to_embed = f"Table: {current_table_name}, Column: {column_name}, Description: {description}"

                # Store additional metadata for filtering or display during retrieval
                # The 'table' field here will contain the specific table name (e.g., 'application_train.csv')
                metadata = {
                    "table": current_table_name,
                    "column": column_name,
                    "original_description": description,  # Keep the full description
                }

                # Create a DataInput object
                data_input = DataInput(text=text_to_embed, metadata=metadata)

                # Use the store_data method from the PineconeManager instance
                response = index_manager.store_data(data_input)

                if response:
                    logging.info(
                        f"Row {idx + 1}: Stored '{column_name}' from table '{current_table_name}'. Chunks: {response.get('chunks')}"
                    )
                else:
                    logging.warning(
                        f"Row {idx + 1}: Failed to store '{column_name}' from table '{current_table_name}'."
                    )
            # --- MODIFICATION END ---

        logging.info("Finished processing and storing all data from CSV.")

    except pd.errors.EmptyDataError:
        logging.error(f"The CSV file '{csv_file_path}' is empty.")
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file '{csv_file_path}': {e}")
    except UnicodeDecodeError as e:
        logging.error(
            f"UnicodeDecodeError: Failed to decode CSV file with 'latin1' encoding. "
            f"It might be encoded differently. Original error: {e}"
        )
        logging.error(
            "Common encodings are 'utf-8', 'latin1', or 'cp1252'. "
            "You might need to open the CSV in a text editor (like Notepad++, VS Code) "
            "and check its encoding, then specify it in pd.read_csv()."
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred during CSV processing: {str(e)}")


if __name__ == "__main__":
    csv_file_path = "data/raw/HomeCredit_columns_description.csv"

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
                    print(
                        f"- Score: {score:.4f} | Table: {table} | Column: {column} | Description: {original_desc}"
                    )
            else:
                print(
                    "No similar descriptions found or an error occurred during search."
                )
        except Exception as e:
            print(f"An error occurred during search: {e}")
