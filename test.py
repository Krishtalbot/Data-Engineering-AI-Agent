# from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
import pandas as pd
from typing import List
import os
import re


class BacklogItem(BaseModel):
    source: List[str] = Field(
        description="List of source tables or files from the Home Credit dataset (e.g., application_train, bureau)"
    )
    transformation: List[str] = Field(
        description="List of ETL operations like joins, filters, or aggregations (e.g., 'join application_train with bureau on SK_ID_CURR', 'filter AMT_INCOME_TOTAL > 50000')"
    )
    business_rules: List[str] = Field(
        description="List of domain-specific rules or conditions (e.g., 'exclude applicants under 18', 'calculate debt-to-income ratio')"
    )
    output: List[str] = Field(
        description="List of output requirements (e.g., 'apply scoring model for loan repayment likelihood 0-100', 'include confidence scores')"
    )


def validate_schema(table_name: str, column_name: str) -> dict:
    try:
        file_path = f"data/raw/{table_name}.csv"
        if not os.path.exists(file_path):
            return {"valid": False, "error": f"File {table_name}.csv not found"}
        df = pd.read_csv(file_path)
        return {"valid": column_name in df.columns, "error": None}
    except Exception as e:
        return {"valid": False, "error": str(e)}


# Initialize LLM
# llm = OllamaLLM(model="llama3:latest", base_url="http://localhost:11434")

api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# Define prompt
prompt = PromptTemplate(
    input_variables=["backlog_item"],
    template="""
You are a data engineering assistant tasked with parsing complex backlog items into structured JSON for data engineering tasks using the Home Credit dataset. Your goal is to accurately extract and categorize components from the backlog item into a JSON object with the keys: "source", "transformation", "business_rules", and "output". Follow these instructions:

1. **Source**: Identify all files, tables, or databases mentioned in the backlog item (e.g., ['application_train', 'bureau', 'previous_application']). Remove file extensions (e.g., use 'application_train' instead of 'application_train.csv'). If no sources are explicitly mentioned, infer likely tables from the Home Credit dataset based on context (e.g., 'applicant data' implies 'application_train').

2. **Transformation**: Extract all processing steps, including joins, filters, aggregations, or column manipulations. Each step should be a clear, standalone string describing the operation. Examples:
   - Joins: 'join application_train with bureau on SK_ID_CURR using inner join'
   - Filters: 'filter AMT_INCOME_TOTAL > 50000'
   - Aggregations: 'calculate average AMT_CREDIT per applicant'
   - Manipulations: 'cast AMT_ANNUITY to float'
   If transformations are implied but not explicit (e.g., 'combine data' implies a join), specify the most likely operation based on context.

3. **Business Rules**: Identify domain-specific logic, conditions, or calculations. Each rule should be a clear string capturing the intent. Examples:
   - Conditions: 'exclude applicants under 18 years old'
   - Calculations: 'calculate debt-to-income ratio as AMT_CREDIT / AMT_INCOME_TOTAL'
   - Domain logic: 'flag applicants with more than 3 previous loans as high-risk'
   If rules are vague, interpret them based on the Home Credit dataset context (e.g., loan repayment likelihood).

4. **Output**: Extract requirements for the final output, including data format, scoring models, or metadata. Each requirement should be a clear string. Examples:
   - 'apply scoring model for loan repayment likelihood with score range 0-100'
   - 'include confidence scores in metadata'
   - 'output results as a table in schema scoring_output'
   If the output format is not specified, default to 'output results as a table'.

5. **Additional Guidelines**:
   - Handle ambiguous or incomplete backlog items by making reasonable assumptions based on the Home Credit dataset (e.g., common columns like SK_ID_CURR, AMT_CREDIT, AMT_INCOME_TOTAL).
   - Ensure column names (e.g., AMT_CREDIT) and table names (e.g., bureau) are accurately identified and preserved in their exact form.
   - If multiple operations are combined in a single sentence, split them into distinct steps or rules for clarity.
   - Exclude any explanatory text, comments, or non-JSON content from the output.
   - Return a valid JSON object with only the keys: "source", "transformation", "business_rules", and "output".

Backlog item:
{backlog_item}
""",
)

# Create parser
parser = JsonOutputParser(pydantic_object=BacklogItem)

# Create LangChain pipeline
backlog_chain = prompt | llm | parser


def parse_backlog_item(backlog_item: str) -> dict | str:
    try:
        # Run LangChain pipeline to parse the backlog item
        parsed_output = backlog_chain.invoke({"backlog_item": backlog_item})

        # Extract columns from transformations and business rules for validation
        columns_to_validate = []
        for item in parsed_output["transformation"] + parsed_output["business_rules"]:
            # Use regex to find potential column names (e.g., AMT_CREDIT, SK_ID_CURR)
            matches = re.findall(r"\b[A-Z_]+\b", item)
            columns_to_validate.extend(matches)

        # Validate source tables
        for table in parsed_output["source"]:
            file_path = f"data/raw/{table}.csv"
            if not os.path.exists(file_path):
                return f"File {table}.csv not found"

        # Validate columns across all source tables
        for column in set(columns_to_validate):
            column_valid = False
            for table in parsed_output["source"]:
                validation_result = validate_schema(table, column)
                if validation_result["valid"]:
                    column_valid = True
                    break
            if not column_valid:
                return f"Column {column} not found in any source table"

        return parsed_output

    except Exception as e:
        return str(e)


# Example usage
input_data = """
Extract applicant data from application_train and bureau tables in the Home Credit dataset. 
Join the tables on SK_ID_CURR using an inner join. 
Filter for applicants with AMT_INCOME_TOTAL greater than 50000. 
Calculate the average AMT_CREDIT per applicant. Apply a business rule to exclude 
applicants under 18 years old and calculate a debt-to-income ratio as AMT_CREDIT 
divided by AMT_INCOME_TOTAL.
"""

parser_result = parse_backlog_item(input_data)
print(parser_result)
