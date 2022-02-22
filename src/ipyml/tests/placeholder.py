try:

    ERROR = False
except ImportError as err:
    ERROR = err


def test_import_regression_base():
    try:
        from ipyml import RegressionBase

        print(RegressionBase)
    except ImportError as err:
        raise err
