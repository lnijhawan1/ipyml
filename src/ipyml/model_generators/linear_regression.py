import ipywidgets as ipyw
import traitlets as trt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

from ipylab import JupyterFrontEnd, Panel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .model_generator import ModelGenerator
from ..utils import TrainedModel
from ..analysis_export import FinalModel


class SKLinearRegression(ModelGenerator):
    
    @trt.default("values")
    def _make_values_dict(self):
        return {
           'Fit Intercept?':self.fit_intercept.value,
        }
    
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        # widgets for parameter selection
        self.fit_intercept = ipyw.Checkbox(value=True, description="Include Intercept?")
        
        selectors = [self.fit_intercept]
        
        for selector in selectors:
            selector.observe(self._update_values, "value")
        
        self.children=[
           self.fit_intercept,
            self.run_button,
        ]
        
    def create_model(self, *_):
        self.run_button.disabled=True

        # Training a model
        X_train, X_test = self._prepare_data()
        target = self.regression_settings.target
        scaler = StandardScaler()
        Xz_train = scaler.fit_transform(X_train.drop([target], axis=1))

        reg = LinearRegression(
            fit_intercept=self.values['Fit Intercept?']
        )
        reg = reg.fit(Xz_train, X_train[target])

        trained_model = TrainedModel(
                regression_config=self.regression_settings,
                train_data=X_train,
                test_data=X_test,
                params=self.values,
                model= reg,
                scaler=scaler,
                model_type='sklearn_LR'
        )
        
        # create model and pop out pane
        self.launch_export_viz(trained_model, 'Fit Linear Regression')
        
        self.run_button.disabled = False





class SMOLS(ModelGenerator):
    
    @trt.default("values")
    def _make_values_dict(self):
        return {
           'Formula':self.formula_input.value,
           'L1 Weight': self.l1_weight.value,
        }
    
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # widgets for parameters
        self.formula_input = ipyw.Text(placeholder="Enter Equation.. (R syntax)")
        self.l1_weight = ipyw.FloatSlider(min=0, max=1, value=.5)

        selectors = [self.formula_input, self.l1_weight]
        
        for selector in selectors:
            selector.observe(self._update_values, "value")
        
        self.children=[
            ipyw.HTML("Available Inputs:"),
            ipyw.HTML(f"{self.regression_settings.inputs}"),
            ipyw.HTML("Select weight for L1 param (If 0-Ridge Regression. If 1-Lasso Regression):"),
            self.l1_weight,
            ipyw.HTML("Target:"),
            ipyw.HTML(f"{self.regression_settings.target}"),
            ipyw.HTML("Enter formula for OLS Regression:"),
            self.formula_input,
            self.run_button,
        ]
        
    def create_model(self, *_):
        self.run_button.disabled=True

        X_train, X_test = self._prepare_data()
        
        # Training a model
        target = self.regression_settings.target
        scaler = StandardScaler()
        inputs = X_train.drop([target], axis=1)
        Xz_train = scaler.fit_transform(inputs)
        Xz_train = pd.DataFrame(Xz_train, index=inputs.index, columns=inputs.columns)
        Xz_train['target'] = X_train['target']
        
        formula = f'{target}~' + self.values['Formula']

        reg = sm.ols(formula, data=Xz_train)
        reg = reg.fit_regularized(L1_wt = self.values['L1 Weight'])

        trained_model = TrainedModel(
                regression_config=self.regression_settings,
                train_data = X_train,
                test_data=X_test,
                params=self.values,
                model= reg,
                scaler=scaler,
                model_type='statsmodel_OLS'
        )
        
        # create final model
        self.launch_export_viz(trained_model, 'Fit OLS Regression')
        
        self.run_button.disabled = False