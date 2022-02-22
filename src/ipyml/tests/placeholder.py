from msilib.schema import Error
from tkinter import E


try:
    from ipyml import RegressionBase  # noqa
    ERROR = False
except ImportError as ERROR:
    pass


def test_import_regression_base():
    if ERROR:
        raise ERROR
