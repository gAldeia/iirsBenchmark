import pytest

import pandas as pd
import numpy  as np
import iirsBenchmark.regressors as regressors

from sklearn.base           import RegressorMixin
from sklearn.exceptions     import NotFittedError
from scipy.optimize         import check_grad
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions     import ConvergenceWarning



ds_names = pd.read_csv(
    './datasets/FeynmanEquations.csv')['Filename'].values


@pytest.mark.parametrize("regressor", regressors.__all__)
def test_has_properties(regressor):
    
    # checking attributes does not need to instantiate the class
    regressor_class = getattr(regressors, regressor)

    assert hasattr(regressor_class, 'stochastic_executions')
    assert hasattr(regressor_class, 'interpretability_spectrum')
    assert hasattr(regressor_class, 'grid_params')


@pytest.mark.parametrize("regressor", regressors.__all__)
def test_is_regressorMixin(regressor):
    
    regressor_class = getattr(regressors, regressor)
    regressor_instance = regressor_class()

    assert isinstance(regressor_instance, RegressorMixin)


@ignore_warnings(category=ConvergenceWarning)
@pytest.mark.parametrize("regressor,ds_name", zip(
    regressors.__all__,
    np.random.choice(ds_names, len(regressors.__all__))
))
def test_fit_and_predict(regressor, ds_name):

    # testing with random datasets. The dataset correctness should be
    # verified in test_feynman.
    
    data = pd.read_csv(
            f'./datasets/train/{ds_name}_UNI.csv', sep=',', 
            header=0, index_col=False).values

    X, y = data[:, :-1], data[:, -1]

    regressor_class = getattr(regressors, regressor)
    # creating with small configurations (only the regressors with
    # slow fitting process)
    predictor = regressor_class(**{
        # ITEA
        'gens' : 10,
        'popsize' : 10, 

        # stop warnings when label is missing
        'labels' : [f'x_{i}' for i in range(X.shape[1])],

        # MLP
        'hidden_layer_sizes' : (50,),
        'activation' : 'identity',
    })
    
    # Should fail, not fitted yet
    with pytest.raises(NotFittedError):
        predictor.predict(X)

    # should succeed
    predictor.fit(X, y)

    predictions = predictor.predict(X)

    assert predictions.shape == (len(y), )
    assert np.any(np.isfinite(predictions))


@pytest.mark.parametrize("regressor,ds_name", zip(
    ['ITEA_regressor', 'Linear_regressor', 'Lasso_regressor'],
    ds_names[:3]
))
def test_gradients(regressor, ds_name):

    data = pd.read_csv(
        f'./datasets/train/{ds_name}_UNI.csv', sep=',', 
        header=0, index_col=False).values

    X, y = data[:, :-1], data[:, -1]
    
    # not using random datasets because ITEA may need a proper configuration
    # to find a solution that does not present discontinuity. The first 3 
    # are simple enough
    regressor_class = getattr(regressors, regressor)
    
    predictor = regressor_class(**{
        # ITEA
        'gens' : 25,
        'popsize' : 25,

        # stop warnings when label is missing
        'labels' : [f'x_{i}' for i in range(X.shape[1])],
    })
    predictor.fit(X, y)

    # auxiliary functions: check_grad takes a single parameter function
    pred_aux = lambda x: predictor.predict(np.array(x).reshape(1, -1))[0]
    grad_aux = lambda x: predictor.gradients(np.array(x).reshape(1, -1))[0]

    for x in X:
        assert check_grad(pred_aux, grad_aux, x, epsilon=1e-4) < np.std(y)


@pytest.mark.parametrize("regressor", regressors.__all__)
def test_to_string(regressor):
    
    regressor_class = getattr(regressors, regressor)
    regressor_instance = regressor_class()

    # methods used in the experiments. All regressors should have at least
    # those methods
    for method in ['fit', 'predict', 'to_str', 'score']:
        # should throw exception if the class does not have to_string
        hasattr(regressor_instance, method)

        assert callable(getattr(regressor_instance, method))
