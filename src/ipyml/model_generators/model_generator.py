import ipywidgets as ipyw
import traitlets as trt
from ipylab import JupyterFrontEnd, Panel
from sklearn.model_selection import train_test_split

from ..analysis_export import FinalModel
from ..utils import RegressionConfiguration


class ModelGenerator(ipyw.VBox):
    values: dict = trt.Dict()
    regression_settings: RegressionConfiguration

    def __init__(self, regression_settings: RegressionConfiguration, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # regression settings from base selection
        self.regression_settings = regression_settings

        # place to store all of the trained models
        self._trained_models = []

        # run button to create the model. always present
        self.run_button = ipyw.Button(description="Create Model")
        self.run_button.on_click(self.create_model)

    def create_model(self, *_):
        """
        Function to create the machine learning models. Will be implemented in the subclasses.
        """
        raise NotImplementedError

    @trt.default("values")
    def _make_values_dict():
        """
        Function to update the values. These will change from class to class so implemented in the subclass.
        """
        raise NotImplementedError

    def _update_values(self, *_):
        self.values = self._make_values_dict()

    def _prepare_data(self):
        """
        This is a universal method that will get the data in the right shape for each algorithm.

        Returns the training and testing data needed.
        """
        dataframe = self.regression_settings.dataframe
        inputs = self.regression_settings.inputs
        target = self.regression_settings.target

        X = dataframe[inputs + [target]]

        if self.regression_settings.validation_feature is None:
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=1)
            return X_train, X_test
        else:
            print("need to implement validation partitioning")
            return

    def launch_export_viz(self, trained_model, name):
        """
        Universal function to launch the finalized models.
        """
        # create model and pop out pane
        final = FinalModel(trained_model=trained_model)
        self._trained_models.append(final)
        panel = Panel(children=[final])
        panel.title.label = name
        app = JupyterFrontEnd()
        app.shell.add(panel, "main", {"mode": "split-left"})

        self.run_button.disabled = False
