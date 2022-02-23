import json

import ipywidgets as ipyw
import matplotlib.pyplot as plt
import pandas as pd
from sklearn_export import Export

from .utils import TrainedModel

class FinalModel(ipyw.VBox):
    def __init__(self, trained_model: TrainedModel, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trained_model = trained_model
        self.scaler = trained_model.scaler

        # create text widgets that describe the parameters of the selected model.
        text_widgets = [
            ipyw.HTML(f"{param}: {trained_model.params[param]}")
            for param in trained_model.params
        ]

        # select available plots, this can be appended in the future
        available_plots = [
            "Predicted vs. Actual",
            "Residuals",
            "Training vs. Validation Residuals",
        ]
        self.plot_selector = ipyw.Dropdown(options=available_plots)
        self.plot_output = ipyw.Output()

        # button to view plot
        self.view_plot_button = ipyw.Button(description="view plot")
        self.view_plot_button.on_click(self.view_plot)

        # export button and file name entry
        self.export_button = ipyw.Button(description="export to json")
        self.export_button.on_click(self.export)
        self.model_name = ipyw.Text(placeholder="Enter file name..")

        self.children = (
            [
                ipyw.HTML("This model was trained with the following parameters:"),
            ]
            + text_widgets
            + [
                ipyw.HTML("Select Plot Type"),
                self.plot_selector,
                ipyw.HBox([self.view_plot_button, self.export_button, self.model_name]),
                self.plot_output,
            ]
        )

    def view_plot(self, *_):
        """
        Function for viewing plots.
        """
        target_name = self.trained_model.regression_config.target
        actual_test = self.trained_model.test_data[target_name]
        actual_train = self.trained_model.train_data[target_name]

        X_test_tr = self.scaler.transform(
            self.trained_model.test_data.drop([target_name], axis=1)
        )
        X_train_tr = self.scaler.transform(
            self.trained_model.train_data.drop([target_name], axis=1)
        )

        X_test_tr = pd.DataFrame(
            X_test_tr,
            index=self.trained_model.test_data.index,
            columns=self.trained_model.test_data.drop([target_name], axis=1).columns,
        )
        X_train_tr = pd.DataFrame(
            X_train_tr,
            index=self.trained_model.train_data.index,
            columns=self.trained_model.train_data.drop([target_name], axis=1).columns,
        )

        predicted_test = self.trained_model.model.predict(X_test_tr)
        predicted_train = self.trained_model.model.predict(X_train_tr)

        min_value = min(min(actual_train), min(predicted_train))
        max_value = max(max(actual_train), max(predicted_train))

        if self.plot_selector.value == "Predicted vs. Actual":
            with self.plot_output:
                self.plot_output.clear_output()
                plt.scatter(
                    actual_test,
                    predicted_test,
                    color="tab:orange",
                    alpha=0.5,
                    label="Test Data",
                )
                plt.scatter(
                    actual_train,
                    predicted_train,
                    color="tab:blue",
                    alpha=0.5,
                    label="Train Data",
                )
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Predicted vs. Actual Values")
                plt.xlim([min_value, max_value])
                plt.ylim([min_value, max_value])
                xpoints = ypoints = plt.xlim()
                plt.legend()
                plt.plot(
                    xpoints,
                    ypoints,
                    linestyle="--",
                    color="k",
                    lw=3,
                    scalex=False,
                    scaley=False,
                )
                plt.show()

        if self.plot_selector.value == "Residuals":
            with self.plot_output:
                self.plot_output.clear_output()
                train_residuals = predicted_train - actual_train
                test_residuals = predicted_test - actual_test
                plt.scatter(
                    predicted_test,
                    test_residuals,
                    color="tab:orange",
                    alpha=0.5,
                    label="Test Data",
                )
                plt.scatter(
                    predicted_train,
                    train_residuals,
                    color="tab:blue",
                    alpha=0.5,
                    label="Train Data",
                )
                plt.xlabel("Predicted Values")
                plt.ylabel("Residual Values")
                plt.title("Residiual Value Plot")
                plt.legend()
                plt.plot(
                    predicted_test,
                    [0 for i in predicted_test],
                    color="k",
                    lw=1,
                    scalex=False,
                    scaley=False,
                )
                plt.show()

        if self.plot_selector.value == "Training vs. Validation Residuals":
            with self.plot_output:
                self.plot_output.clear_output()
                train_residuals = predicted_train - actual_train
                test_residuals = predicted_test - actual_test
                plt.hist(
                    test_residuals, color="tab:orange", alpha=0.5, label="Test Data"
                )
                plt.hist(
                    train_residuals, color="tab:blue", alpha=0.5, label="Train Data"
                )
                plt.xlabel("Residual Values")
                plt.ylabel("Count")
                plt.legend()
                plt.show()

    # TODO: This will need to change for non-scikit learn functions. conditional statements or better way?
    def export(self, *_):
        """
        Function to export model to json.
        """
        if self.trained_model.model_type == "statsmodel_OLS":
            target_col = self.trained_model.regression_config.target
            saved_params = self.trained_model.model.params.to_dict()
            saved_formula = f"{target_col} ~ " + "+".join(
                self.trained_model.model.params.index
            ).replace(
                "Intercept", "1"
            )  # (is this necessary?)

            model_data = {
                "weights": saved_params,
                "target_column": target_col,
                "formula": saved_formula,
            }
            model_data["model_type"] = self.trained_model.model_type
            g = open(f"{self.model_name.value}.json", "w")
            json.dump(model_data, g)
            g.close()

        else:
            # Exporting the model
            self.export_button.disabled = True
            export = Export([self.scaler, self.trained_model.model])
            scaler_data = export.template[0].load_model_data()
            model_data = export.template[1].load_model_data()
            model_data.update(scaler_data)

            # lets get the min and max of each inputs
            bounds = {}
            for input in self.trained_model.regression_config.inputs:
                input_series = self.trained_model.train_data[input]
                bounds[input] = (min(input_series), max(input_series))

            # Opening JSON file
            g = open(f"{self.model_name.value}.json", "w")
            model_data["inputs"] = list(self.trained_model.regression_config.inputs)
            model_data["target"] = self.trained_model.regression_config.target
            model_data["bounds"] = bounds
            model_data["model_type"] = self.trained_model.model_type
            model_data = dict(model_data)
            json.dump(model_data, g)
            self.export_button.disabled = False
