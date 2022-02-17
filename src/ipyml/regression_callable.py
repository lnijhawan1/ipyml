import numpy as np


def relu(z):
    z = np.array(z)
    for subarray in z:
        if type(subarray) == np.float64:
            continue
        subarray[subarray<0] = 0
    return z

def tanh(z):
    z = np.array(z)
    for subarray in z:
        np.tanh(subarray)
    
    return z
    
def sigmoid(z):
    z = np.array(z)
    for subarray in z:
        subarray = 1/(1 + np.exp(-subarray))
    
    return z
    
def softmax(z):
    z = np.array(z)
    for subarray in z:
        subarray = np.exp(subarray)/np.sum(np.exp(subarray), axis=1, keepdims=True)
    
    return z

def standard_scaler(X, model_data):
    mean = np.asarray(model_data['mean'])
    std = np.asarray(model_data['std'])
    Xz = (X-mean)/std
    return Xz


ACTIVATION_FUNCTIONS = {
    'logistic':sigmoid,
    'softmax':softmax,
    'relu':relu,
    'tanh':tanh,
    'identity':lambda z: z
}

def run_neural_net(model: dict, **kwargs):
    inputs = [
        kwargs[var_name]
        for var_name in model["inputs"]
    ]
    # Output  and hidden activations
    output_activation = ACTIVATION_FUNCTIONS.get(model['output_activation'], lambda z: z)
    hidden_activation = ACTIVATION_FUNCTIONS.get(model['hidden_activation'], lambda z: z)

    layer_values = np.array(
        [
            item if isinstance(item, (list, np.ndarray)) else np.array([item])
            for item in inputs
        ]
    ).T

    layer_values = standard_scaler(layer_values, model)
    output_layer_index = len(model["weights"]) - 1
    for layer_index, (layer_weights, layer_biases, num_rows, num_cols) in enumerate(
        zip(model['weights'], model['bias'], model['numRows'], model['numColumns'])
    ): 
        weights = np.asarray(layer_weights).reshape(num_rows, num_cols, order='F')
        bias = np.asarray(layer_biases)
        hidden_layer = np.dot(layer_values, weights) + bias
        if layer_index == output_layer_index:
            layer_values = output_activation(hidden_layer)
        else:
            layer_values = hidden_activation(hidden_layer)
    if len(layer_values) == 1:
        return layer_values[0]
    else:
        return layer_values.flatten()
    
    

def run_linear_regression(model: dict, **kwargs):
    inputs = [
       kwargs[var_name]
        for var_name in model["inputs"]
    ]
    
    X = np.array(
        [
            item if isinstance(item, (list, np.ndarray)) else np.array([item])
            for item in inputs
        ]
    ).T
    X = standard_scaler(X, model)
    coefs = np.asarray(model['coefficients'])
    intercepts = np.asarray(model['intercept'])
    h = np.dot(X, coefs) + intercepts
    return h