# from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from typing import List, Optional, Literal
import os
import re

from src.vectorDB.pinecone_conf import (
    PineconeManager,
    DataInput,
)  # Adjust path if needed
from dotenv import load_dotenv


load_dotenv()


class Source(BaseModel):
    name: str = Field(
        ..., description="Name of the source table or file (e.g., 'application_train')"
    )
    description: Optional[str] = Field(
        None, description="A brief description of the source."
    )


class DataQualityCheck(BaseModel):
    expectation_type: str = Field(
        ...,
        description="The Great Expectations expectation type (e.g., 'expect_column_values_to_not_be_null')",
    )
    column: Optional[str] = Field(
        None, description="The name of the column this expectation applies to."
    )
    kwargs: Optional[dict] = Field(
        None,
        description="Dictionary of keyword arguments for the expectation (e.g., {'min_value': 0, 'max_value': 100}).",
    )
    severity: Literal["CRITICAL", "WARNING"] = Field(
        "CRITICAL",
        description="Severity of the check: 'CRITICAL' (pipeline halts) or 'WARNING' (alerts, but pipeline continues).",
    )
    description: Optional[str] = Field(
        None, description="A human-readable description of this data quality check."
    )


# New: Define a model for detailed transformation operations
class TransformationOperation(BaseModel):
    operation: str = Field(
        ...,
        description="Specific operation within a transformation (e.g., 'impute_mean', 'filter_rows', 'rename_column')",
    )
    columns: Optional[List[str]] = Field(
        None, description="List of columns affected by this operation."
    )
    data_type: Optional[str] = Field(
        None,
        description="Expected data type after transformation (e.g., 'numeric', 'categorical', 'date').",
    )
    details: Optional[str] = Field(
        None, description="More specific details about this operation."
    )


class TransformationStep(BaseModel):
    type: str = Field(
        ...,
        description="Type of transformation (e.g., 'join', 'filter', 'aggregation', 'column_manipulation')",
    )
    details: str = Field(
        ...,
        description="High-level detailed description of the transformation operation (e.g., 'join application_train with bureau on SK_ID_CURR', 'handle missing values in all columns').",
    )
    output_table_name: Optional[str] = Field(
        None,
        description="Suggested name for the PySpark DataFrame after this transformation (e.g., 'joined_df', 'cleaned_data').",
    )
    transformations_details: Optional[List[TransformationOperation]] = Field(
        None,
        description="A list of granular operations within this transformation step, with specific columns and types.",
    )
    data_quality_checks: Optional[List[DataQualityCheck]] = Field(
        None,
        description="List of Great Expectations checks to apply after this transformation step.",
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
        description="Detailed description of the output requirement (e.g., 'apply scoring model for loan repayment likelihood with score range 0-100', 'output results as a table').",
    )
    input_table_name: Optional[str] = Field(
        None,
        description="Name of the PySpark DataFrame that serves as input for this output step.",
    )
    output_table_name: Optional[str] = Field(
        None,
        description="Suggested name for the final PySpark DataFrame or destination after this output step (e.g., 'scored_data').",
    )
    data_quality_checks: Optional[List[DataQualityCheck]] = Field(
        None,
        description="List of Great Expectations checks to apply to the output of this step.",
    )


class BacklogItem(BaseModel):
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
    global_data_quality_checks: Optional[List[DataQualityCheck]] = Field(
        None,
        description="List of Great Expectations checks that apply to the overall final dataset.",
    )


# Remove pandas import if not used elsewhere, or modify this function
def validate_schema(table_name: str, column_name: str) -> dict:
    if column_desc_pinecone_manager is None:
        return {
            "valid": False,
            "error": "PineconeManager for column descriptions not initialized.",
        }

    try:
        # Step 1: Encode the query text to get a vector
        # The query text can still be semantic to help guide the search,
        # but the metadata filter will ensure precision.
        query_text = f"description for column {column_name} in table {table_name}"
        query_vector = column_desc_pinecone_manager.model.encode(query_text).tolist()

        # Step 2: Define the metadata filter
        # CRUCIAL: Ensure the table_name includes '.csv' if that's how it's stored in Pinecone metadata
        # As per your latest store_column_descriptions.py, it should be like 'application_train.csv'
        filter_condition = {
            "table": {
                "$eq": f"{table_name}.csv"
            },  # Example: {"table": {"$eq": "application_train.csv"}}
            "column": {
                "$eq": column_name
            },  # Example: {"column": {"$eq": "SK_ID_CURR"}}
        }

        # Step 3: Directly query the Pinecone index using the filter
        results = column_desc_pinecone_manager.index.query(
            vector=query_vector,
            top_k=1,  # We only need to know if at least one exact match exists
            include_metadata=True,  # Include metadata to confirm (optional, but good for debugging)
            filter=filter_condition,  # THIS IS THE KEY ADDITION
        )

        if results and results.matches:
            # If any match is returned after applying the filter, it means an exact entry was found.
            # No need for manual `best_match.metadata.get(...) == ...` checks, as the filter already did that.
            # You can optionally check `best_match.score` here if you want to ensure a minimum semantic relevance
            # even for exact metadata matches, but for schema validation, the filter's match is usually sufficient.
            return {"valid": True, "error": None}
        else:
            return {
                "valid": False,
                "error": f"Column '{column_name}' not found in table '{table_name}.csv' in Pinecone (no exact metadata match).",
            }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Error during Pinecone schema validation: {str(e)}",
        }


api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

column_desc_index_name = "csv-description"
try:
    column_desc_pinecone_manager = PineconeManager(column_desc_index_name)
except Exception as e:
    print(f"Failed to initialize PineconeManager for column descriptions: {e}")
    # Decide how to handle this critical error: exit, or proceed with limited functionality
    column_desc_pinecone_manager = None

# Define prompt
prompt = PromptTemplate(
    input_variables=["backlog_item"],
    template="""
You are a highly detailed and intelligent data engineering assistant. Your primary goal is to parse complex natural language backlog items into an extremely precise, structured JSON format for automated data engineering tasks. You must infer as much detail as possible, especially regarding data quality expectations and specific transformation operations.

**Home Credit Dataset Context:** Assume you are working with the Home Credit loan dataset, which includes tables like `application_train`, `bureau`, `previous_application`, etc., and common columns such as `SK_ID_CURR`, `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `DAYS_BIRTH`, `NAME_CONTRACT_TYPE`, `TARGET`, `LOAN_REPAYMENT_SCORE`. Use this context to make intelligent inferences when details are not explicitly stated.

**JSON Schema Requirements:**
Generate a JSON object with the following top-level keys: "source", "transformation", "business_rules", "output", and "global_data_quality_checks". Each should contain a list of objects, structured as follows:

1.  **Source**:
    * Identify all files, tables, or databases. Remove file extensions.
    * Infer likely tables from the Home Credit dataset if not explicit (e.g., 'applicant data' implies 'application_train').
    * Fields: `name` (string), `description` (optional string).
    * Example: `{{"name": "application_train", "description": "Main applicant data"}}`

2.  **Transformation**:
    * Extract all ETL processing steps.
    * Fields:
        * `type` (string, e.g., 'join', 'filter', 'aggregation', 'column_manipulation').
        * `details` (string): High-level description of the operation.
        * `output_table_name` (optional string): A logical name for the PySpark DataFrame *after* this transformation (e.g., 'joined_df', 'cleaned_data'). Infer if not given.
        * `transformations_details` (optional list of objects): For `column_manipulation` types, break down into specific operations.
            * Each object: `operation` (string, e.g., 'impute_mean', 'filter_rows', 'rename_column'), `columns` (optional list of strings), `data_type` (optional string, infer if possible: 'numeric', 'categorical', 'date'), `details` (optional string).
            * Example for `handle missing values`: `{{"operation": "impute_mean", "columns": ["AMT_INCOME_TOTAL", "AMT_CREDIT"], "data_type": "numeric", "details": "Impute missing numeric values with column mean."}}`
        * `data_quality_checks` (optional list of `DataQualityCheck` objects): Great Expectations checks *after* this transformation.
            * Each `DataQualityCheck` object: `expectation_type` (string, e.g., `expect_column_values_to_not_be_null`), `column` (optional string), `kwargs` (optional dict for parameters like `min_value`, `max_value`, `value_set`, `regex`), `severity` (string: "CRITICAL" or "WARNING"), `description` (optional string).
            * **Crucial Inference for Data Quality:** Based on the transformation details and Home Credit context, infer common and critical data quality checks.
                * **Post-Join:** `expect_column_values_to_not_be_null` on join keys, `expect_column_pair_values_to_be_unique`.
                * **General Numerical/Date Columns:** `expect_column_values_to_be_between`, `expect_column_values_to_match_datetime_format`.
                * **Categorical Columns:** `expect_column_distinct_values_to_be_in_set`.
                * **IDs:** `expect_column_values_to_be_unique`.
                * Use `CRITICAL` for essential checks (e.g., nulls on IDs, out-of-range crucial numerical values) and `WARNING` for less critical but important checks (e.g., unexpected distributions).

3.  **Business Rules**:
    * Identify domain-specific logic or conditions.
    * Fields: `type` (string, e.g., 'condition', 'calculation', 'domain_logic'), `details` (string).
    * Example: `{{"type": "condition", "details": "exclude applicants under 18 years old"}}`

4.  **Output**:
    * Extract final output requirements.
    * Fields:
        * `type` (string, e.g., 'scoring_model', 'format', 'metadata').
        * `details` (string): High-level description.
        * `input_table_name` (optional string): Name of the PySpark DataFrame serving as input.
        * `output_table_name` (optional string): Name for the final output DataFrame/destination.
        * `data_quality_checks` (optional list of `DataQualityCheck` objects): GE checks *on the final output*.
            * **For Scoring Models:** Always include `expect_column_values_to_be_between` for the score (0-100), and `expect_column_values_to_be_of_type`. Consider `expect_column_proportion_of_values_to_be_between` for predicted classes/scores.

5.  **Global Data Quality Checks (New Section)**:
    * Use `global_data_quality_checks` (optional list of `DataQualityCheck` objects) for overall dataset checks (e.g., `expect_table_row_count_to_be_between` on the final output).

**General Guidelines for LLM Generation:**
* **Infer everything:** If a detail isn't explicitly mentioned, infer it based on common data engineering practices and the Home Credit dataset context.
* **Be specific:** For `expectation_type` and `operation` fields, use precise terms.
* **Valid JSON:** Ensure the entire output is a syntactically correct JSON object.
* **No extraneous text:** Only output the JSON.

Backlog item:
{backlog_item}
""",
)

parser = JsonOutputParser(pydantic_object=BacklogItem)

backlog_chain = prompt | llm | parser


def parse_backlog_item(backlog_item: str) -> dict | str:
    try:
        parsed_output = backlog_chain.invoke({"backlog_item": backlog_item})

        all_mentioned_columns = set()  # Use a set to store all unique columns mentioned
        generated_columns = set()  # New set to store columns expected to be generated

        # Helper function to extract columns from details string
        def extract_columns_from_text(text: str) -> List[str]:
            # This regex is simple and might catch non-column words.
            # A more robust solution might involve a lexicon of common column names
            # from the Home Credit dataset or a more sophisticated NLP model.
            return re.findall(r"\b[A-Z_]+\b", text)

        # --- Collect all mentioned columns and identify generated ones ---

        # 1. From Transformation Steps
        for item in parsed_output.get("transformation", []):
            if "details" in item and isinstance(item["details"], str):
                all_mentioned_columns.update(extract_columns_from_text(item["details"]))
            if item.get("transformations_details"):
                for op_detail in item["transformations_details"]:
                    if op_detail.get("columns"):
                        all_mentioned_columns.update(op_detail["columns"])
            if item.get("data_quality_checks"):
                for dq_check in item["data_quality_checks"]:
                    if dq_check.get("column"):
                        all_mentioned_columns.add(dq_check["column"])

        # 2. From Business Rules
        for item in parsed_output.get("business_rules", []):
            if "details" in item and isinstance(item["details"], str):
                all_mentioned_columns.update(extract_columns_from_text(item["details"]))
            # You might expand this section if business rules explicitly define new calculated columns
            # For example: if item.get("type") == "calculation" and "new_column" in item.get("details"):
            #    generated_columns.add("new_column")

        # 3. From Output Requirements (Crucial for identifying generated columns like LOAN_REPAYMENT_SCORE)
        for item in parsed_output.get("output", []):
            if "details" in item and isinstance(item["details"], str):
                all_mentioned_columns.update(extract_columns_from_text(item["details"]))
            if item.get("data_quality_checks"):
                for dq_check in item["data_quality_checks"]:
                    if dq_check.get("column"):
                        all_mentioned_columns.add(dq_check["column"])
                        # If this output step involves a scoring model, its output column is generated
                        if item.get("type") == "scoring_model":
                            generated_columns.add(
                                dq_check["column"]
                            )  # Mark as generated!

        # 4. From Global Data Quality Checks
        if parsed_output.get("global_data_quality_checks"):
            for dq_check in parsed_output["global_data_quality_checks"]:
                if dq_check.get("column"):
                    all_mentioned_columns.add(dq_check["column"])
        source_tables = []
        for source_item in parsed_output.get("source", []):
            if "name" in source_item and isinstance(source_item["name"], str):
                file_path = f"data/raw/{source_item['name']}.csv"
                if not os.path.exists(file_path):
                    return f"File {source_item['name']}.csv not found"
                source_tables.append(source_item["name"])
            else:
                print(
                    f"Warning: 'name' key missing or not a string in source item: {source_item}"
                )
        columns_for_source_validation = all_mentioned_columns - generated_columns

        for column in columns_for_source_validation:
            column_found_in_any_source = False
            for source_table_name in source_tables:
                validation_result = validate_schema(source_table_name, column)
                if validation_result["valid"]:
                    column_found_in_any_source = True
                    break
            if not column_found_in_any_source:
                return f"Column {column} not found in any source table"

        return parsed_output

    except Exception as e:
        return str(e)
