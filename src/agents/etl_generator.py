from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from pathlib import Path
import os
import json
import sys
from backlog_parser import parse_backlog_item


def generate_etl_code(backlog_item):
    parsed_result = parse_backlog_item(backlog_item)
    if isinstance(parsed_result, str):
        return f"Error parsing backlog item: {parsed_result}"

    # Setup AutoGen agents for code generation
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

    completion_signal = "FINISH"
    generated_code = []

    def is_termination_msg_with_capture(msg):
        content = msg.get("content", "")
        if completion_signal in content:
            return True

        if msg.get("role") == "assistant":
            # Simple heuristic to extract code blocks
            code_blocks = []
            in_code_block = False
            current_block = []

            for line in content.split("\n"):
                if line.strip().startswith("```python") or line.strip().startswith(
                    "```pyspark"
                ):
                    in_code_block = True
                    continue
                elif line.strip() == "```" and in_code_block:
                    in_code_block = False
                    if current_block:
                        code_blocks.append("\n".join(current_block))
                        current_block = []
                    continue

                if in_code_block:
                    current_block.append(line)

            if code_blocks:
                generated_code.extend(code_blocks)

        return False

    user_proxy_agent = UserProxyAgent(
        name="User",
        code_execution_config={"executor": code_executor},
        is_termination_msg=is_termination_msg_with_capture,
        human_input_mode="NEVER",
    )

    system_message = """
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
    """

    spark_coder = AssistantAgent(
        name="Spark code generator",
        system_message=system_message,
        llm_config=llm_config,
    )

    json_input = json.dumps(parsed_result)

    user_proxy_agent.initiate_chat(
        spark_coder,
        message=f"Generate PySpark code for this parsed backlog item: {json_input}",
    )

    if generated_code:
        return "\n\n".join(generated_code)
    else:
        return "No code was generated. There may have been an error in the generation process."


def save_code_to_file(code, filename):
    filepath = Path("spark_code_store") / filename
    with open(filepath, "w") as f:
        f.write(code)
    return filepath


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python etl_generator.py "<backlog item description>" [output_filename]'
        )
        print(
            'Example: python etl_generator.py "Combine application_train and bureau tables" etl_script.py'
        )
        return

    backlog_item = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else "generated_etl_code.py"

    print(f"Generating ETL code for: {backlog_item}")
    generated_code = generate_etl_code(backlog_item)

    if not generated_code.startswith("Error"):
        filepath = save_code_to_file(generated_code, output_filename)
        print(f"Code successfully generated and saved to {filepath}")
    else:
        print(generated_code)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        test_backlog = """
        Combine application_train, previous_application, and credit_card_balance on SK_ID_CURR. 
        Flag high-risk applicants (those with AMT_CREDIT > 1M or DAYS_EMPLOYED < 365). 
        Output a risk score (0-100).
        """
        print(f"Using test backlog item: {test_backlog}")
        generated_code = generate_etl_code(test_backlog)
        save_code_to_file(generated_code, "test_etl_code.py")
        print(f"Test code saved to spark_code_store/test_etl_code.py")
    else:
        main()
