import ipywidgets as ipyw
import pandas as pd
import traitlets as trt
from IPython.display import display
from ipylab import JupyterFrontEnd, Panel
from utils import RegressionConfiguration
from model_generators import NeuralNetwork, SKLinearRegression, SMOLS


class RegressionBase(ipyw.VBox):   
    regression_configuration = trt.Instance(RegressionConfiguration)

    @trt.default('regression_configuration')
    def _make_regression_config(self):
        return RegressionConfiguration(
            dataframe=self.dataframe,
            inputs=list(self.inputs_select.value),
            target= self.target_select.value,
            validation_feature= self.validation_column_select.value,
            validation_value= self.validation_column_value.value,
        )

    def __init__(self, dataframe: pd.DataFrame):
        super().__init__()
        # dataframe to be used for ML
        self.dataframe = dataframe

        # create our buttons for each type of model desired
        self.nn_button = ipyw.Button(layout=dict(width='40px', height='40px'), tooltip='Create Neural Network', icon='bolt')
        self.nn_button.on_click(self.create_nn)

        self.lr_button = ipyw.Button(layout=dict(width='40px', height='40px'), tooltip='Create Linear Regression', icon='line-chart')
        self.lr_button.on_click(self.create_lr)

        self.ols_button = ipyw.Button(layout=dict(width='40px', height='40px'), tooltip='Create OLS Regression', icon='plus-square-o')
        self.ols_button.on_click(self.create_ols)
        
        # button to view the dataframe if desired
        self.view_dataframe_button = ipyw.Button(description='view dataframe',layout=dict(width='250px', height='40px'), tooltip='View Dataframe', icon='bars')
        self.view_dataframe_button.on_click(self.view_dataframe)
        
        # keep our generated models for reference
        self._generated_models = []
        columns = list(self.dataframe.columns)
        
        # create the dropdowns that power the widget
        self.target_select = ipyw.Dropdown(options=columns)
        self.inputs_select = ipyw.SelectMultiple(options=columns)
        self.validation_column_select = ipyw.Dropdown(options=columns + [None], value=None)
        self.validation_column_select.observe(self._update_column_set, 'value')
        self.validation_column_value = ipyw.Dropdown()

        selectors = [self.target_select, self.inputs_select, self.validation_column_select]
        for selector in selectors:
            selector.observe(self._update_selectors, 'value')
        
        self._update_selectors()
        self.make_children()
        
    def _update_column_set(self, *_):
        '''
        Updates the set of possible options in the validation column.
        '''
        column = self.validation_column_select.value
        if not column:
            self.validation_column_value.options = []
            return
        all_values = set(list(self.dataframe[column]))
        self.validation_column_value.options = all_values
        
    def view_dataframe(self, *_):
        '''
        Function to view the dataframe.
        '''
        output = ipyw.Output()
        with output:
            display(self.dataframe)
        panel = Panel(children=[output])
        panel.title.label = "DataFrame Sample"
        app = JupyterFrontEnd()
        app.shell.add(panel, "main", {"mode": "split-right"})

    def create_ols(self, *_):
        '''
        Function to create OLS regression loader.
        '''
        self._make_regression_config()
        model = SMOLS(regression_settings=self.regression_configuration)
        self.create_generator(model, "OLS Regression Configuration")

    def create_lr(self, *_):
        '''
        Function to create ordinary linear regression loader.
        '''
        self._make_regression_config()
        SKLR = SKLinearRegression(regression_settings=self.regression_configuration)
        self.create_generator(SKLR, "Linear Regression Configuration")

    def create_nn(self, *_):
        '''
        Function to create the neural network loader.
        '''
        self._make_regression_config()
        NN = NeuralNetwork(regression_settings=self.regression_configuration)
        self.create_generator(NN, 'Neural Network Configuration')


    def create_generator(self, generator, name):
        '''
        Create the panel needed to run models.
        '''
        self._generated_models.append(generator)
        panel = Panel(children=[generator])
        panel.title.label = name
        app = JupyterFrontEnd()
        app.shell.add(panel, "main", {"mode": "split-right"})
        
    def make_children(self):
        '''
        Creates the children for the base selector.
        '''
        self.children = [
                self.view_dataframe_button,
                ipyw.HTML('Select Target Feature'),
                self.target_select,
                ipyw.HTML('Select Inputs for Model'),
                self.inputs_select,
                ipyw.HTML('Select Validation Column'),
                self.validation_column_select,
                ipyw.HTML('Select Training Value'),
                self.validation_column_value,
                ipyw.HBox(children=[self.nn_button, self.lr_button, self.ols_button]),
            ]

    def _update_selectors(self, *_):
        '''
        SUPER hacky placehodler method until we get a more established way to prevent 
        'cross contamination' of dropdowns, so to say...
        '''
        all_columns = set(list(self.dataframe.columns))
        target_value = self.target_select.value
        input_values = self.inputs_select.value
        validation_value = self.validation_column_select.value
        
        if validation_value is not None:
            valid_target_values = all_columns - set(input_values) - set([validation_value,])
            valid_input_values = all_columns - set([target_value,]) - set([validation_value,])
            valid_validation_values = all_columns - set([target_value,]) - set(input_values)

        else:
            valid_target_values = all_columns - set(input_values)
            valid_input_values = all_columns - set([target_value,]) 
            valid_validation_values = all_columns - set([target_value,]) - set(input_values)           

        dropdowns = {
            self.target_select:(target_value, valid_target_values), 
            self.inputs_select:(input_values, valid_input_values),
            self.validation_column_select:(validation_value, valid_validation_values)    
        }

        for dropdown in dropdowns:
            dropdown.unobserve(self._update_selectors, "value")
        
        with self.hold_trait_notifications():
            for dropdown in dropdowns:
                if dropdown == self.target_select:
                    dropdown.options = valid_target_values
                    if target_value in valid_target_values:
                        dropdown.value=target_value
                    else:
                        dropdown.value = valid_target_values[0]

                if dropdown == self.inputs_select:
                    dropdown.options = valid_input_values
                    new_values = []
                    for col in input_values:
                        if col in dropdown.options:
                            new_values.append(col)
                    dropdown.value = new_values

                if dropdown == self.validation_column_select:
                    if validation_value is None:
                        dropdown.options = [None] + list(valid_validation_values)
                        dropdown.value = None
                    else:
                        dropdown.options = valid_validation_values
                        if validation_value in valid_validation_values:
                            dropdown.value = validation_value
                        else:
                            dropdown.value = None

        for dropdown in dropdowns:
            dropdown.observe(self._update_selectors, "value")
