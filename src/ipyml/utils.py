from dataclasses import dataclass

import ipywidgets as ipyw
import pandas as pd
import traitlets as trt


@dataclass
class RegressionConfiguration:
    dataframe: pd.DataFrame
    inputs: list
    target: str
    validation_feature: str
    validation_value: trt.Any()


@dataclass
class TrainedModel:
    regression_config: RegressionConfiguration
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    params: dict
    model_type: str
    model: trt.Any()  # need to change this, just don't know what to call it exactly
    scaler: trt.Any()  # need to change this, future problem


class HiddenLayers(ipyw.VBox):
    number_of_layers = trt.Int(default_value=0)
    value = trt.Tuple()

    def make_children(self):
        children = [ipyw.HTML("Select Layer Sizes")]
        for i in range(self.number_of_layers):
            remove_button = ipyw.Button(icon="minus", layout=dict(width="40px"))
            remove_button.on_click(self.remove_layer)
            children.append(ipyw.HTML("Select Size of Layer {}".format(i + 1)))
            children.append(ipyw.HBox(children=[self.layer_sizes[i], remove_button]))

        self.children = children + [self.add_layer_button]

    def __init__(self):
        super().__init__()
        self.add_layer_button = ipyw.Button(icon="plus", layout=dict(width="40px"))
        self.add_layer_button.on_click(self.add_layer)
        self.layer_sizes = []
        self.add_layer()

    def add_layer(self, *_):
        self.number_of_layers += 1
        new_layer = ipyw.Text(value="100", layout=dict(width="80px"))
        new_layer.observe(self.get_layers, "value")
        self.layer_sizes.append(new_layer)
        self.make_children()

    def remove_layer(self, *_):
        self.number_of_layers -= 1
        self.make_children()

    def get_layers(self, *_):
        hidden_layers = tuple()
        for child in self.children:
            if isinstance(child, ipyw.HBox):
                hidden_layers += (int(child.children[0].value),)
        self.value = hidden_layers
        return hidden_layers


class DataFrameViewer(ipyw.VBox):
    dataframe = trt.Instance(pd.DataFrame)
    col_selector = trt.Instance(ipyw.Dropdown)
    output = trt.Instance(ipyw.Output)

    def __init__(self, dataframe, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataframe = dataframe
