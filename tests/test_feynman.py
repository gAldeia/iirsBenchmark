import pandas as pd
import numpy  as np
import pytest

from scipy.optimize     import check_grad
from sklearn.exceptions import NotFittedError

from iirsBenchmark.groundtruth                   import Feynman_regressor
from iirsBenchmark.groundtruth._FeynmanEquations import feynmanPyData


# It is important that this test module have a comprehensive validation of
# the synthetic data.

ds_names = pd.read_csv(
    './datasets/Feynman/FeynmanEquations.csv')['Filename'].values


@pytest.mark.parametrize("ds_name", ds_names)
def test_feynman_functions_evals_correctly_on_train(ds_name):
    train_data = pd.read_csv(
            f'./datasets/Feynman/train/{ds_name}_UNI.csv', sep=',', 
            header=0, index_col=False).values

    X_train, y_train = train_data[:, :-1], train_data[:, -1]

    predictor = Feynman_regressor(
        equation_name=ds_name)
        
    # Should fail, not fitted yet
    with pytest.raises(NotFittedError):
        predictor.predict(X_train)

    predictor.fit(X_train, y_train)

    # Train data was not generated by me. This assertion checks if
    # the predictions on this data is close enought to the provided real values.
    # If this assert fails, then the original expressions are incorrect.
    # We'll use only 100 observations to have faster tests
    assert np.allclose(
        predictor.predict(X_train[:100, :]),
        y_train[:100], rtol=1e-04, atol=1e-04)


@pytest.mark.parametrize("ds_name", ds_names)
def test_feynman_functions_evals_correctly_on_test(ds_name):
   
    test_data = pd.read_csv(
            f'./datasets/Feynman/test/{ds_name}_LHS.csv', sep=',',
            header=0, index_col=False).values

    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    predictor = Feynman_regressor(
        equation_name=ds_name)
        
    # Should fail, not fitted yet
    with pytest.raises(NotFittedError):
        predictor.predict(X_test)
        
    predictor.fit(X_test, y_test)

    # LHS data was generated by me. If this assert fails, then the generated
    # data is invalid.
    assert np.allclose(
        predictor.predict(X_test), y_test, rtol=1e-04, atol=1e-04)


@pytest.mark.parametrize("ds_name", ds_names)
def test_feynman_autodiff_gradients_on_test(ds_name):
    
    test_data = pd.read_csv(
            f'./datasets/Feynman/test/{ds_name}_LHS.csv', sep=',', 
            header=0, index_col=False).values

    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    predictor = Feynman_regressor(
        equation_name=ds_name).fit(X_test, y_test)

    # auxiliary functions: check_grad takes a single parameter function
    pred_aux = lambda x: predictor.predict(np.array(x).reshape(1, -1))[0]
    grad_aux = lambda x: predictor.gradients(np.array(x).reshape(1, -1))[0]

    # If the 2 previous tests didn't fail, then the gradients can be tested
    # with train or test data. Let's use test data, since it has less samples
    # (100) than the test data (1000).
    for x in X_test:
        assert check_grad(pred_aux, grad_aux, x, epsilon=1e-4) < np.std(y_test)


@pytest.mark.parametrize("ds_name", ds_names)
def test_feynman_to_string(ds_name):
    
    predictor = Feynman_regressor(
        equation_name=ds_name)

    assert predictor.to_str() == \
           feynmanPyData[ds_name]['string expression']