from backlog_parser import parse_backlog_item


def test():
    input_data = """
Combine application_train, previous_application, and credit_card_balance on SK_ID_CURR. Flag high-risk applicants (those with AMT_CREDIT > 1M or DAYS_EMPLOYED < 365). Output a risk score (0-100).    """
    result = parse_backlog_item(input_data)
    print(result)


if __name__ == "__main__":
    test()
