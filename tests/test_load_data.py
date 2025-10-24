from app.load_data import load_data


def test_load_data_structure():
    """
    Basic unit test to ensure that the data loading function works properly.
    It checks that:
    - the returned DataFrame is not empty,
    - and that key columns ('category', 'return_rate') are present.
    """
    df = load_data()

    # The DataFrame should contain data
    assert not df.empty, "DataFrame should not be empty."

    # Essential columns must exist
    for col in ["category", "return_rate"]:
        assert col in df.columns, f"Missing column: {col}"
