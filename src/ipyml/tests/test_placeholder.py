def test_import_regression_base():
    try:
        from ipyml.api import RegressionBase

        print(RegressionBase)
    except ImportError as err:
        raise err
