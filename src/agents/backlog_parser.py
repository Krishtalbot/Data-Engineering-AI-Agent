# from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
import pandas as pd
from typing import List, Optional, Literal
import os
import re


class Source(BaseModel):
    """Defines a source table or file from the Home Credit dataset."""

    name: str = Field(
        ..., description="Name of the source table or file (e.g., 'application_train')"
    )
    description: Optional[str] = Field(
        None, description="A brief description of the source."
    )


class TransformationStep(BaseModel):
    """Defines a single ETL transformation operation."""

    type: str = Field(
        ...,
        description="Type of transformation (e.g., 'join', 'filter', 'aggregation', 'column_manipulation')",
    )
    details: str = Field(
        ...,
        description="Detailed description of the transformation operation (e.g., 'join application_train with bureau on SK_ID_CURR', 'filter AMT_INCOME_TOTAL > 50000')",
    )


class BusinessRule(BaseModel):
    """Defines a single domain-specific business rule or condition."""

    type: str = Field(
        ...,
        description="Type of business rule (e.g., 'condition', 'calculation', 'domain_logic')",
    )
    details: str = Field(
        ...,
        description="Detailed description of the business rule (e.g., 'exclude applicants under 18 years old', 'calculate debt-to-income ratio as AMT_CREDIT / AMT_INCOME_TOTAL')",
    )


class OutputRequirement(BaseModel):

    type: str = Field(
        ...,
        description="Type of output requirement (e.g., 'scoring_model', 'format', 'metadata')",
    )
    details: str = Field(
        ...,
        description="Detailed description of the output requirement (e.g., 'apply scoring model for loan repayment likelihood with score range 0-100', 'output results as a table')",
    )


class BacklogItem(BaseModel):
    """
    Represents a structured backlog item for data engineering tasks,
    parsed from a complex natural language description.
    """

    source: List[Source] = Field(
        description="List of source tables or files from the Home Credit dataset."
    )
    transformation: List[TransformationStep] = Field(
        description="List of ETL operations like joins, filters, or aggregations."
    )
    business_rules: List[BusinessRule] = Field(
        description="List of domain-specific rules or conditions."
    )
    output: List[OutputRequirement] = Field(
        description="List of output requirements (e.g., 'apply scoring model for loan repayment likelihood 0-100', 'include confidence scores')."
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
You are a data engineering assistant tasked with parsing complex backlog items into structured JSON for data engineering tasks using the Home Credit dataset. Your goal is to accurately extract and categorize components from the backlog item into a JSON object with the keys: "source", "transformation", "business_rules", and "output". Each of these keys should contain a list of objects, following the detailed schema provided below.

Follow these instructions for each section:

1.  **Source**:
    * Identify all files, tables, or databases mentioned in the backlog item (e.g., 'application_train', 'bureau').
    * Remove file extensions (e.g., use 'application_train' instead of 'application_train.csv').
    * If no sources are explicitly mentioned, infer likely tables from the Home Credit dataset based on context (e.g., 'applicant data' implies 'application_train').
    * Format each source as an object with a "name" key (string) and an optional "description" key (string).
    * Example: `{{ "name": "application_train", "description": "Main applicant data" }}`

2.  **Transformation**:
    * Extract all processing steps, including joins, filters, aggregations, or column manipulations.
    * Each step should be formatted as an object with a "type" key (string, e.g., 'join', 'filter', 'aggregation', 'column_manipulation') and a "details" key (string).
    * The "details" should be a clear, standalone string describing the operation.
    * Examples:
        * Join: `{{ "type": "join", "details": "join application_train with bureau on SK_ID_CURR using inner join" }}`
        * Filter: `{{ "type": "filter", "details": "filter AMT_INCOME_TOTAL > 50000" }}`
        * Aggregation: `{{ "type": "aggregation", "details": "calculate average AMT_CREDIT per applicant" }}`
        * Manipulation: `{{ "type": "column_manipulation", "details": "cast AMT_ANNUITY to float" }}`
    * If transformations are implied but not explicit (e.g., 'combine data' implies a join), specify the most likely operation based on context.

3.  **Business Rules**:
    * Identify domain-specific logic, conditions, or calculations.
    * Each rule should be formatted as an object with a "type" key (string, e.g., 'condition', 'calculation', 'domain_logic') and a "details" key (string).
    * The "details" should be a clear string capturing the intent.
    * Examples:
        * Condition: `{{ "type": "condition", "details": "exclude applicants under 18 years old" }}`
        * Calculation: `{{ "type": "calculation", "details": "calculate debt-to-income ratio as AMT_CREDIT / AMT_INCOME_TOTAL" }}`
        * Domain Logic: `{{ "type": "domain_logic", "details": "flag applicants with more than 3 previous loans as high-risk" }}`
    * If rules are vague, interpret them based on the Home Credit dataset context (e.g., loan repayment likelihood).

4.  **Output**:
    * Extract requirements for the final output, including data format, scoring models, or metadata.
    * Each requirement should be formatted as an object with a "type" key (string, e.g., 'scoring_model', 'format', 'metadata') and a "details" key (string).
    * Examples:
        * Scoring Model: `{{ "type": "scoring_model", "details": "apply scoring model for loan repayment likelihood with score range 0-100" }}`
        * Metadata: `{{ "type": "metadata", "details": "include confidence scores in metadata" }}`
        * Format: `{{ "type": "format", "details": "output results as a table in schema scoring_output" }}`
    * If the output format is not specified, default to `{{ "type": "format", "details": "output results as a table" }}`.

5.  **Additional Guidelines**:
    * Handle ambiguous or incomplete backlog items by making reasonable assumptions based on the Home Credit dataset (e.g., common columns like SK_ID_CURR, AMT_CREDIT, AMT_INCOME_TOTAL).
    * Ensure column names (e.g., AMT_CREDIT) and table names (e.g., bureau) are accurately identified and preserved in their exact form.
    * If multiple operations are combined in a single sentence, split them into distinct steps or rules for clarity.
    * Exclude any explanatory text, comments, or non-JSON content from the output.
    * Return a valid JSON object.
Backlog item:
{backlog_item}
""",
)

# Create parser
parser = JsonOutputParser(pydantic_object=BacklogItem)

# Create LangChain pipeline
backlog_chain = prompt | llm | parser


# ... (your existing imports and Pydantic models)


def parse_backlog_item(backlog_item: str) -> dict | str:
    try:
        parsed_output = backlog_chain.invoke({"backlog_item": backlog_item})

        columns_to_validate = []
        # Access the 'details' key of each dictionary
        for item in parsed_output["transformation"]:
            # Ensure 'details' key exists before accessing
            if "details" in item and isinstance(item["details"], str):
                matches = re.findall(r"\b[A-Z_]+\b", item["details"])
                columns_to_validate.extend(matches)
            else:
                # Handle cases where 'details' might be missing or not a string
                # You might log this or raise a more specific error
                print(
                    f"Warning: 'details' key missing or not a string in transformation item: {item}"
                )

        for item in parsed_output["business_rules"]:
            # Ensure 'details' key exists before accessing
            if "details" in item and isinstance(item["details"], str):
                matches = re.findall(r"\b[A-Z_]+\b", item["details"])
                columns_to_validate.extend(matches)
            else:
                print(
                    f"Warning: 'details' key missing or not a string in business rule item: {item}"
                )

        # The rest of your code remains the same, but now accessing dictionary keys for 'source'
        for source_item in parsed_output["source"]:
            # Ensure 'name' key exists before accessing
            if "name" in source_item and isinstance(source_item["name"], str):
                file_path = f"data/raw/{source_item['name']}.csv"
                if not os.path.exists(file_path):
                    return f"File {source_item['name']}.csv not found"
            else:
                print(
                    f"Warning: 'name' key missing or not a string in source item: {source_item}"
                )
                # Decide how to handle invalid source entries, e.g., skip or raise error

        for column in set(columns_to_validate):
            column_valid = False
            for source_item in parsed_output["source"]:
                if "name" in source_item and isinstance(source_item["name"], str):
                    validation_result = validate_schema(source_item["name"], column)
                    if validation_result["valid"]:
                        column_valid = True
                        break
            if not column_valid:
                return f"Column {column} not found in any source table"

        return parsed_output

    except Exception as e:
        return str(e)
