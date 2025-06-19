from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from pathlib import Path
import os
import json
import sys
import re

from src.agents.backlog_parser import parse_backlog_item
from src.vectorDB.pinecone_conf import *  # Assuming this correctly imports store_data and DataInput

generated_code = []
llm_config = None


load_dotenv()
logging.basicConfig(level=logging.INFO)  # Ensure logging is configured

# --- Configuration for your ETL Pinecone Index ---
# Define the Pinecone index name specifically for ETL generated data
# You can change 'etl-generated-data' to any name you prefer for this specific data.
etl_pinecone_index_name = "user-input"
etl_pinecone_manager = None  # Initialize as None
pc_api = os.getenv("PINECONE_API_KEY")
pc_client = Pinecone(api_key=pc_api)
etl_pinecone_manager = PineconeManager(etl_pinecone_index_name)


def generate_etl_code(parsed_backlog_dict: dict, original_backlog_item_str: str):
    global generated_code
    generated_code.clear()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable not set", parsed_backlog_dict

    config_list = [
        {
            "model": "gemini-2.0-flash",
            "api_type": "google",
            "stream": False,
            "api_key": api_key,
        }
    ]

    global llm_config
    llm_config = {"config_list": config_list}

    workdir = Path("spark_code_store")
    workdir.mkdir(exist_ok=True)
    code_executor = LocalCommandLineCodeExecutor(work_dir=workdir)

    def is_termination_msg(msg):
        # A message indicates termination if it contains "FINISH"
        return "FINISH" in msg.get("content", "")

    user_proxy_agent = UserProxyAgent(
        name="User",
        code_execution_config={"executor": code_executor},
        is_termination_msg=is_termination_msg,
        human_input_mode="NEVER",  # Set to NEVER for automated execution during agent interaction
    )

    system_message_coder = """
    You are a Spark code generator who writes PySpark code. You will write
    clean, efficient PySpark code from a JSON specification that you will receive. The JSON
    will contain detailed information about sources, transformations (including granular operations and
    data quality checks), business rules, and output requirements (including model scoring and
    post-scoring data quality checks).

    Your code should:
    1.  **Import necessary libraries**: Include `pyspark.sql`, `pyspark.sql.functions`, `great_expectations`, and `great_expectations.dataset.sparkdf_dataset`.
    2.  **Set up a Spark session**: Create a SparkSession.
    3.  **Read source data**: Read each source data file from "data/raw/{source_name}.csv" into a PySpark DataFrame.
    4.  **Initialize Great Expectations**:
        * Get a Great Expectations Data Context (use `gx.get_context()`).
        * For each section with `data_quality_checks` (e.g., post-join, post-cleaning, post-scoring, global):
            * Create a unique `ExpectationSuite` name (e.g., `f"{dataframe_name}_suite"`).
            * Create the `ExpectationSuite` using `context.create_expectation_suite()`.
            * Iterate through the `data_quality_checks` list. For each check, add the corresponding expectation to the suite using `suite.add_expectation(gx.expectations.Expect...())`. Use the provided `expectation_type` and `kwargs`.
            * Save the expectation suite using `context.save_expectation_suite()`.
            * Get a Great Expectations Data Context (use `gx.get_context()`).
    * For each section with `data_quality_checks`:
        * **Crucially, when managing Expectation Suites:**
            * To *create or get* an Expectation Suite by name, use `suite = context.create_expectation_suite(suite_name_string)`. This method is idempotent; it will create the suite if it doesn't exist, or return the existing one.
            * To *save* changes to an Expectation Suite (after adding expectations), use `context.save_expectation_suite(suite)`.

        * Create a unique `ExpectationSuite` name (e.g., `f"{dataframe_name}_suite"`).
        * Create or get the `ExpectationSuite` using `suite = context.create_expectation_suite(your_suite_name)`.
        * Iterate through the `data_quality_checks` list. For each check, add the corresponding expectation to the suite using `suite.add_expectation(gx.expectations.Expect...())`. Use the provided `expectation_type` and `kwargs`.
        * **After adding all expectations for a suite, remember to save it:** `context.save_expectation_suite(suite)`.

    5.  **Apply Transformations**:
        * Implement each transformation step in the order they appear.
        * For `join` transformations, use `df_source.join(df_other, on=join_key, how=join_type)`. Use the `output_table_name` for the new DataFrame variable name.
        * For `column_manipulation`, iterate through `transformations_details` and apply specific PySpark operations (e.g., `.fillna()`, `.withColumn()`, `.cast()`).
        * **Crucially, after each transformation step that has `data_quality_checks` defined:**
            * Convert the current PySpark DataFrame to a Great Expectations `SparkDFDataset` (e.g., `ge_df = SparkDFDataset(current_df)`).
            * Load the relevant `ExpectationSuite` (using the suite name you created).
            * Run validation: `validation_results = ge_df.validate(expectation_suite=your_suite)`.
            * **Implement Error Handling for GE**:
                * Check `validation_results["success"]`.
                * If `False`:
                    * Print a clear message indicating which check failed.
                    * Iterate through `validation_results["results"]` to log specific failures (`expectation_config`, `result`).
                    * If `severity` is "CRITICAL", raise an `Exception` to halt the pipeline (`raise Exception("Critical data quality check failed.")`).
                    * If `severity` is "WARNING", just log the issue (do not halt).
                * Always call `context.build_data_docs()` after validation to generate reports.
    6.  **Implement Business Rules**: Apply filters, aggregations, or derive new columns based on the business rules.
    7.  **Apply AutoML Scoring Model**:
        * When `output.type` is "scoring_model", include a placeholder for applying the model. Since the model itself is external, assume it's a function or a loaded Spark MLlib pipeline.
        * Example Placeholder: `scored_df = your_automl_model_function(transformed_df) # Replace with actual model application`
        * Add columns for `LOAN_REPAYMENT_SCORE` and any metadata (e.g., `confidence_score` if the model provides it).
        * Apply `data_quality_checks` to the `scored_df` similarly to transformations, with appropriate error handling.
    8.  **Apply Global Data Quality Checks**: If `global_data_quality_checks` are present, apply them to the final output DataFrame before writing.
    9.  **Write Output Data**: Write the final DataFrame (e.g., `scored_applicants_df`) to a specified location (e.g., Parquet file) as per the output format.

    **Code Structure and Best Practices:**
    * Use clear variable names for DataFrames (e.g., `application_train_df`, `bureau_df`, `joined_df`, `cleaned_df`, `scored_df`).
    * Add appropriate comments explaining each section, transformation, and data quality check.
    * Ensure the code is syntactically correct and executable.

    IMPORTANT: After you've generated the code, respond with "FINISH" at the end of your message.
    The next agent will optimize this code.
    """

    system_message_optimizer = """
    You are a senior data engineer specializing in PySpark optimization. You will receive PySpark code that has been
    auto-generated based on a set of sources, transformations, business rules, **Great Expectations data quality checks**,
    and **AutoML model application**.

    Your job is to review and improve the code for:
    1.  **Performance**: Optimize Spark operations (e.g., efficient joins - consider broadcast joins if appropriate, repartitioning, caching DataFrames before multiple uses or before expensive validation steps).
    2.  **Readability and Structure**: Refactor into modular functions for clarity, ensure clean formatting, and consistent variable naming.
    3.  **Spark Best Practices**: Ensure correct usage of Spark APIs, avoid common pitfalls (e.g., unnecessary shuffles, excessive data collection to driver), and manage data types effectively.
    4.  **Error Handling**: Review and enhance existing `try-except` blocks, especially around file I/O and Great Expectations validation.
    5.  **Scalability**: Ensure the pipeline is designed to handle large datasets gracefully.
    6. **Great Expectations & AutoML Integration**:
    * Ensure that GE validation steps are correctly integrated, optimized (e.g., validation on cached DataFrames), and that the AutoML model application placeholder is robust.
    * **Specifically, verify that Expectation Suites are created using `context.create_expectation_suite(suite_name)` and saved using `context.save_expectation_suite(suite_object)` after expectations are added.** Do NOT alter the logic of the GE expectations themselves, only their execution context and correct API usage.

    You should:
    -   Refactor the code where needed, not just suggest improvements.
    -   Maintain exact functionality, only correcting bugs or enhancing performance/readability.
    -   Improve comments to be concise and informative.
    -   Ensure the optimized code remains executable and logically correct, including all GE validation and AutoML model application steps.

    At the end of your response, include a summary of the key improvements made, followed by the keyword "FINISH".
    """

    spark_coder = AssistantAgent(
        name="Spark_code_generator",
        system_message=system_message_coder,
        llm_config=llm_config,
    )
    spark_optimizer = AssistantAgent(
        name="Spark_code_optimizer",
        system_message=system_message_optimizer,
        llm_config=llm_config,
    )

    json_input = json.dumps(parsed_backlog_dict)

    coder_chat_result = user_proxy_agent.initiate_chat(
        spark_coder,
        message=f"Generate PySpark ETL code based on the following JSON specification: {json_input}",
    )

    generated_code_by_coder = ""
    last_coder_message = spark_coder.last_message()
    if last_coder_message and "content" in last_coder_message:
        content = last_coder_message["content"]
        code_blocks = re.findall(
            r"```(?:python|pyspark)\n(.*?)\n```", content, re.DOTALL
        )
        if code_blocks:
            generated_code_by_coder = "\n\n".join(code_blocks)
        else:
            # Fallback if code blocks are not explicitly formatted but content looks like code
            if (
                "from pyspark.sql" in content
                or "spark = SparkSession.builder" in content
            ):
                generated_code_by_coder = content

    if not generated_code_by_coder:
        return (
            "No initial code was generated by the coder or could not be extracted.",
            parsed_backlog_dict,
        )

    optimizer_chat_result = user_proxy_agent.initiate_chat(
        spark_optimizer,
        message=f"Please optimize the following PySpark code:\n{generated_code_by_coder}",
    )

    optimized_code = ""
    last_optimizer_message = spark_optimizer.last_message()
    if last_optimizer_message and "content" in last_optimizer_message:
        content = last_optimizer_message["content"]
        code_blocks = re.findall(
            r"```(?:python|pyspark)\n(.*?)\n```", content, re.DOTALL
        )
        if code_blocks:
            optimized_code = "\n\n".join(code_blocks)
        else:
            if (
                "from pyspark.sql" in content
                or "spark = SparkSession.builder" in content
            ):
                optimized_code = content

    if optimized_code:
        return optimized_code, parsed_backlog_dict
    else:
        return (
            "No optimized code was generated or extracted. There may have been an error in the optimization process.",
            parsed_backlog_dict,
        )


def save_code_to_file(code, filename):
    filepath = Path("spark_code_store") / filename
    with open(filepath, "w") as f:
        f.write(code)
    return filepath


def main():
    if len(sys.argv) < 2:
        backlog_item_str = """
        Generate an ETL pipeline to predict loan default. 
        Use application_train and bureau tables. Join them on SK_ID_CURR. 
        Handle all missing values, then apply our loan default scoring model (scores 0-100).
        Ensure the final score is never null and is between 0 and 100. 
        Also, check that the joined data has unique SK_ID_CURR after joining.
        """
        print(f"Using test backlog item: {backlog_item_str}")

    else:
        backlog_item_str = sys.argv[1]

    output_filename = sys.argv[2] if len(sys.argv) > 2 else "generated_etl_code.py"

    parsed_json_output = parse_backlog_item(backlog_item_str)

    if isinstance(parsed_json_output, str):
        print(f"Error parsing backlog item: {parsed_json_output}")
        return

    # --- Step 2: Human-in-the-loop verification for Pinecone storage ---
    human_verifier = UserProxyAgent(
        name="Human_Verifier",
        human_input_mode="ALWAYS",  # This agent ALWAYS asks for human input
        max_consecutive_auto_reply=1,
        code_execution_config=False,  # No code execution for this agent
    )

    verification_message = f"""
    I have processed the backlog item and generated a parsed JSON output.
    Please review the proposed data for storage in Pinecone.

    Original Backlog Item:
    {backlog_item_str}

    Parsed JSON Output:
    {json.dumps(parsed_json_output, indent=2)}

    Do you approve storing this data in Pinecone? Please respond with 'yes' or 'no'.
    """

    reviewer = AssistantAgent(
        name="Reviewer",
        llm_config=llm_config,
        is_termination_msg=lambda x: x.get("content", "").strip().lower()
        in ["yes", "y", "no", "n", "exit"],
    )
    chat_result = reviewer.initiate_chat(
        human_verifier,
        message=verification_message,
        silent=False,
    )

    user_consent = reviewer.last_message()["content"].strip().lower()

    if user_consent in ["yes", "y"]:
        if etl_pinecone_manager:  # Check if the manager was successfully initialized
            try:
                # backlog_item_str and parsed_json_output are assumed to be defined earlier
                # json.dumps converts the dict to a string for storage in Pinecone's text field
                message = etl_pinecone_manager.store_data(
                    DataInput(
                        text=f"problem_type::{backlog_item_str} => {json.dumps(parsed_json_output, indent=2)}",
                        metadata={
                            "source": "autogen_parsed_etl",
                            "backlog_item": backlog_item_str,
                        },  # Add more relevant metadata here
                    )
                )
                if message:
                    print(f"Stored parsed result in Pinecone: {message}")
                else:
                    print("Failed to store the parsed result in Pinecone.")
            except Exception as e:  # Catch any exceptions during the storage call
                print(f"Error storing data to Pinecone using ETL manager: {e}")
        else:
            print(
                "Skipping data storage to Pinecone: ETL Pinecone Manager not initialized due to previous errors."
            )
    else:
        print("Parsed data not stored in Pinecone as per user's choice.")

    # --- Step 3: Proceed with ETL code generation and optimization (regardless of Pinecone consent) ---
    print("\n--- Generating and optimizing ETL code ---")
    final_code, _ = generate_etl_code(parsed_json_output, backlog_item_str)

    if final_code.startswith("Error"):
        print(f"Error generating/optimizing code: {final_code}")
        return

    filepath = save_code_to_file(final_code, output_filename)
    print(f"Optimized code successfully saved to {filepath}")


if __name__ == "__main__":
    main()
