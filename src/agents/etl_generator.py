from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from pathlib import Path
import os
import json
import sys
import re
from src.agents.backlog_parser import parse_backlog_item
from src.vectorDB.pinecone_conf import *

generated_code = []


def generate_etl_code(backlog_item):
    global generated_code
    generated_code.clear()
    parsed_result = parse_backlog_item(backlog_item)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable not set"

    config_list = [
        {
            "model": "gemini-2.0-flash",
            "api_type": "google",
            "stream": False,
            "api_key": api_key,
        }
    ]

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
        human_input_mode="NEVER",  # Set to NEVER for automated execution
    )

    system_message_coder = """
    You are a Spark code generator who writes PySpark code. You will write
    clean, efficient PySpark code from a JSON that you will receive. In the JSON you will
    get sources, transformation and business rules. You will analyze them
    all and write a fully functioning PySpark code.

    Your code should:
    1. Import necessary libraries
    2. Set up a Spark session
    3. Read the source data files from "data/raw/{source}.csv"
    4. Apply all transformations from the JSON
    5. Implement all business rules from the JSON
    6. Write the output data according to the output specifications
    7. Include appropriate comments explaining each section

    IMPORTANT: After you've generated the code, respond with "FINISH" at the end of your message.
    The next agent will optimize this code. Make sure the code is syntactically correct.
    """

    system_message_optimizer = """
    You are a senior data engineer specializing in PySpark optimization. You will receive PySpark code that has been
    auto-generated based on a set of sources, transformations, and business rules.

    Your job is to review and improve the code for:
    1. Performance (e.g., use of caching, partitioning, avoiding shuffles)
    2. Readability and structure (e.g., modular functions, clean formatting)
    3. Spark best practices (e.g., limiting wide transformations, efficient joins, correct data types)
    4. Error handling (e.g., try-except blocks for file reads/writes)
    5. Scalability (e.g., repartitioning before large joins, using broadcast joins appropriately)

    You should:
    - Refactor the code where needed, not just suggest improvements.
    - Keep functionality exactly the same unless there's a bug.
    - Improve comments but keep it short.
    - Ensure the code remains executable and logically correct.

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

    json_input = json.dumps(parsed_result)

    # First chat: User asks spark_coder to generate code
    # The result of initiate_chat is returned, but for extracting code,
    # it's often more reliable to get the last message directly from the agent.
    coder_chat_result = user_proxy_agent.initiate_chat(
        spark_coder,
        message=f"Generate PySpark ETL code based on the following JSON specification: {json_input}",
    )

    # Extract the generated code from the coder's last message
    generated_code_by_coder = ""
    last_coder_message = spark_coder.last_message()
    if last_coder_message and "content" in last_coder_message:
        content = last_coder_message["content"]
        # Use regex to find all code blocks enclosed by ```python or ```pyspark
        code_blocks = re.findall(
            r"```(?:python|pyspark)\n(.*?)\n```", content, re.DOTALL
        )
        if code_blocks:
            generated_code_by_coder = "\n\n".join(code_blocks)
        else:
            # Fallback: if no markdown code blocks found from coder
            if (
                "from pyspark.sql" in content
                or "spark = SparkSession.builder" in content
            ):
                generated_code_by_coder = (
                    content  # Assume it's code if it contains Spark-related keywords
                )

    if not generated_code_by_coder:
        return "No initial code was generated by the coder or could not be extracted."

    # Second chat: User asks spark_optimizer to optimize the generated code
    # Pass the code generated by the coder to the optimizer
    optimizer_chat_result = user_proxy_agent.initiate_chat(
        spark_optimizer,
        message=f"Please optimize the following PySpark code:\n{generated_code_by_coder}",
    )

    # Extract the optimized code from the optimizer's last message
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
            # Fallback: if no markdown code blocks found from optimizer
            if (
                "from pyspark.sql" in content
                or "spark = SparkSession.builder" in content
            ):
                optimized_code = content

    if optimized_code:
        return optimized_code
    else:
        return "No optimized code was generated or extracted. There may have been an error in the optimization process."


def save_code_to_file(code, filename):
    filepath = Path("spark_code_store") / filename
    with open(filepath, "w") as f:
        f.write(code)
    return filepath


def main():
    if len(sys.argv) < 2:
        # Default test backlog item if no arguments are provided
        test_backlog = """
        Generate an ETL pipeline to predict loan default on the main application table,
        joining with previous credit bureau data and ensuring all missing values are handled.
        """
        print(f"Using test backlog item: {test_backlog}")
        final_code = generate_etl_code(test_backlog)

        if not final_code.startswith("Error"):
            # Assuming store_data and DataInput are available from pinecone_conf
            try:
                message = store_data(
                    DataInput(
                        text=f"problem_type::{test_backlog} => {final_code}",
                        metadata={"source": "autogen_optimized_etl"},
                    )
                )
                if message:
                    print(f"Stored optimized result: {message}")
                else:
                    print("Failed to store the optimized result.")
            except NameError:
                print(
                    "Warning: store_data or DataInput not found. Skipping data storage."
                )
            except Exception as e:
                print(f"Error storing data to Pinecone: {e}")

            # Save the optimized code to a file
            save_code_to_file(final_code, "optimized_test_etl_code.py")
            print(
                f"Optimized code saved to spark_code_store/optimized_test_etl_code.py"
            )
            # print(f"Optimized Generated Code:\n{final_code}") # Uncomment to print
        else:
            print(final_code)  # Print the error message
    else:
        # Use command-line arguments if provided
        backlog_item = sys.argv[1]
        output_filename = sys.argv[2] if len(sys.argv) > 2 else "generated_etl_code.py"

        print(f"Generating and optimizing ETL code for: {backlog_item}")
        final_code = generate_etl_code(backlog_item)

        if not final_code.startswith("Error"):
            filepath = save_code_to_file(final_code, output_filename)
            print(f"Code successfully generated, optimized, and saved to {filepath}")
            # Assuming store_data and DataInput are available from pinecone_conf
            # try:
            #     message = store_data(
            #         DataInput(
            #             text=f"problem_type::{backlog_item} => {final_code}",
            #             metadata={"source": "autogen_optimized_etl"},
            #         )
            #     )
            #     if message:
            #         print(f"Stored optimized result: {message}")
            #     else:
            #         print("Failed to store the optimized result.")
            # except NameError:
            #     print(
            #         "Warning: store_data or DataInput not found. Skipping data storage."
            #     )
            # except Exception as e:
            #     print(f"Error storing data to Pinecone: {e}")
        else:
            print(final_code)


if __name__ == "__main__":
    main()
