from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from ipylab import JupyterFrontEnd, Panel

import traitlets as trt
import ipywidgets as ipyw
from utils import TrainedModel, HiddenLayers
from analysis_export import FinalModel
from .model_generator import ModelGenerator

class NeuralNetwork(ModelGenerator):
    hidden_layers: HiddenLayers = trt.Instance(HiddenLayers, args=())

    
    @trt.default("values")
    def _make_values_dict(self):
        return {
           'Max Iterations':self.max_iteration.value,
           'Hidden Layer Configuration':self.hidden_layers.value,
           'Activation Function':self.activation.value,
           'Solver':self.solver.value, 
        }
    
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # set up the parameter selection tools
        self.max_iteration = ipyw.IntSlider(min=1000, max=10000, description='Max Iter:')
        self.activation = ipyw.Dropdown(options=['identity', 'logistic', 'tanh', 'relu'], value='relu', description='Activation:')
        self.solver = ipyw.Dropdown(options=['lbfgs', 'sgd', 'adam'], value='adam', description='Solver:')
        
        selectors = [self.hidden_layers, self.max_iteration, self.activation, self.solver]
        
        # add observers
        for selector in selectors:
            selector.observe(self._update_values, "value")
        
        # set children
        self.children= [
           self.hidden_layers,
            self.max_iteration,
            self.activation,
            self.solver,
            self.run_button,
        ]
        
    def create_model(self, *_):
        # initial set up to get values
        self.run_button.disabled=True

        # Training a model
        X_train, X_test = self._prepare_data()
        target = self.regression_settings.target
        scaler = StandardScaler()
        Xz_train = scaler.fit_transform(X_train.drop([target], axis=1))

        reg = MLPRegressor(
            hidden_layer_sizes=self.values['Hidden Layer Configuration'], 
            max_iter=self.values['Max Iterations'],
            solver=self.values['Solver'],
            activation=self.values['Activation Function'],
            verbose=1
        )
        reg = reg.fit(Xz_train, X_train[target])

        trained_model = TrainedModel(
                regression_config=self.regression_settings,
                train_data=X_train,
                test_data=X_test,
                params=self.values,
                model= reg,
                scaler=scaler,
                model_type='sklearn_MLP'
        )
        
        # create final model and pop out pane
        self.launch_export_viz(trained_model, 'Fit Neural Network')