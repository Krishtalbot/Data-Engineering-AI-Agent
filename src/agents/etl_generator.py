from backlog_parser import parse_backlog_item


def test():
    input_data = """
    Extract applicant data from application_train and bureau tables in the Home Credit dataset. 
    Join the tables on SK_ID_CURR using an inner join. 
    Filter for applicants with AMT_INCOME_TOTAL greater than 50000. 
    Calculate the average AMT_CREDIT per applicant. Apply a business rule to exclude 
    applicants under 18 years old and calculate a debt-to-income ratio as AMT_CREDIT 
    divided by AMT_INCOME_TOTAL.
    """
    result = parse_backlog_item(input_data)
    print(result)
