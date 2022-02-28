from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ipyml.utils import RegressionConfiguration

try:
    from ipyml.api import RegressionBase
    from ipyml.model_generators import SMOLS, NeuralNetwork, SKLinearRegression
except ImportError as err:
    raise err


def test_import_regression_base():

    diabetes = load_diabetes(return_X_y=False, as_frame=True)["frame"]
    regr = RegressionBase(diabetes)
    neural_network(regr)
    linear_regression(regr)


def neural_network(regr_base: RegressionBase):
    setup = regr_base
    setup.regression_configuration = RegressionConfiguration(
        dataframe=regr_base.dataframe,
        inputs=["bp", "s1", "s2", "s3", "s4", "s5", "s6", "bmi", "age", "sex"],
        target="target",
        validation_feature=None,
        validation_value=None,
    )

    nn = NeuralNetwork(regression_settings=setup.regression_configuration)

    # lets give our neural network some parameters, then we have to test vs. sklearn
    nn.activation.value = "relu"
    nn.solver.value = "adam"
    nn.max_iteration.value = 5000
    nn.hidden_layers.value = (10, 10, 10)

    # now compute OUR neural network
    # Training a model
    X_train, X_test = nn._prepare_data()
    target = nn.regression_settings.target
    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.drop([target], axis=1))

    reg = MLPRegressor(
        hidden_layer_sizes=nn.values["Hidden Layer Configuration"],
        max_iter=nn.values["Max Iterations"],
        solver=nn.values["Solver"],
        activation=nn.values["Activation Function"],
        random_state=1,
    )
    reg = reg.fit(Xz_train, X_train[target])

    # now compute what the sklearn model SHOULD be
    correct_model = MLPRegressor(
        hidden_layer_sizes=(10, 10, 10),
        activation="relu",
        solver="adam",
        max_iter=5000,
        random_state=1,
    ).fit(Xz_train, X_train[target])

    # how can we compare this 'correct' fitted model with our fitted model? predictions and attributes
    assert correct_model.n_features_in_ == reg.n_features_in_
    assert correct_model.n_iter_ == reg.n_iter_

    # lets make predictions
    sk_predictions = correct_model.predict(
        [[0 for _ in range(correct_model.n_features_in_)]]
    )
    our_predictions = reg.predict([[0 for _ in range(reg.n_features_in_)]])

    assert sk_predictions[0] == our_predictions[0]


def linear_regression(regr_base: RegressionBase):
    setup = regr_base
    setup.regression_configuration = RegressionConfiguration(
        dataframe=regr_base.dataframe,
        inputs=["bp", "s1", "s2", "s3", "s4", "s5", "s6", "bmi", "age", "sex"],
        target="target",
        validation_feature=None,
        validation_value=None,
    )

    lr = SKLinearRegression(regression_settings=setup.regression_configuration)
    # parameters
    lr.fit_intercept.value = True

    # now compute OUR neural network
    # Training a model
    X_train, X_test = lr._prepare_data()
    target = lr.regression_settings.target
    scaler = StandardScaler()
    Xz_train = scaler.fit_transform(X_train.drop([target], axis=1))

    reg = LinearRegression(fit_intercept=lr.values["Fit Intercept?"])
    reg = reg.fit(Xz_train, X_train[target])

    # now compute what the sklearn model SHOULD be
    correct_model = LinearRegression(fit_intercept=True).fit(Xz_train, X_train[target])

    # how can we compare this 'correct' fitted model with our fitted model? predictions and attributes
    assert correct_model.n_features_in_ == reg.n_features_in_
    assert correct_model.intercept_ == reg.intercept_

    # lets make predictions
    sk_predictions = correct_model.predict(
        [[0 for _ in range(correct_model.n_features_in_)]]
    )
    our_predictions = reg.predict([[0 for _ in range(reg.n_features_in_)]])

    assert sk_predictions[0] == our_predictions[0]
