import pytest

import pandas as pd
import numpy as np

import iirsBenchmark.explainers as explainers
from iirsBenchmark.metrics import (
    stability, jaccard_stability, infidelity, neighborhood)

from iirsBenchmark.feynman    import Feynman_regressor
from iirsBenchmark.exceptions import NotApplicableException


ds_names = pd.read_csv(
    './datasets/FeynmanEquations.csv')['Filename'].values


def load_samples_by_ds_name(ds_name):
    data = pd.read_csv(
        f'./datasets/test/{ds_name}_LHS.csv', sep=',', 
        header=0, index_col=False).values

    # Using only the first 100 observations of the data
    X, y = data[:100, :-1], data[:100, -1]

    return X, y


@pytest.mark.parametrize("explainer,ds_name", zip(
    explainers.__all__,
    ds_names[:len(explainers.__all__)]
))
def test_stability(explainer, ds_name):
    # We'll explain the original equation and check if the stability value
    # is within the expected range, and how it deals with nans.

    X, y = load_samples_by_ds_name(ds_name)    

    explainer_instance = getattr(explainers, explainer)(
        predictor=Feynman_regressor(equation_name=ds_name).fit(X, y)
    )
    
    # Testing only for applicable explainers
    if (explainer_instance.agnostic == True or \
        'Feynman_regressor' in explainer_instance.agnostic) and \
        (explainer_instance.local_scope == True):
            
            explainer_instance.fit(X, y)

            # We'll explain 10 instances and see if all passes on the tests
            stabilities = np.array([stability(
                explainer_instance.explain_local,
                X[i].reshape(1, -1),
                neighborhood(X[i].reshape(1, -1), X,
                    factor=0.001, size=5),
            ) for i in range(10)])

            assert all(stabilities >= 0.0)
            assert all(np.isfinite(stabilities))


@pytest.mark.parametrize("explainer,ds_name", zip(
    explainers.__all__,
    ds_names[:len(explainers.__all__)]
))
def test_jaccard_stability(explainer, ds_name):
    # We'll explain the original equation and check if the stability value
    # is within the expected range, and how it deals with nans.

    X, y = load_samples_by_ds_name(ds_name)    

    explainer_instance = getattr(explainers, explainer)(
        predictor=Feynman_regressor(equation_name=ds_name).fit(X, y)
    )
    
    # Testing only for applicable explainers
    if (explainer_instance.agnostic == True or \
        'Feynman_regressor' in explainer_instance.agnostic) and \
        (explainer_instance.local_scope == True):
            
            explainer_instance.fit(X, y)

            # We'll explain 10 instances and see if all passes on the tests
            jaccards = np.array([jaccard_stability(
                explainer_instance.explain_local,
                X[i].reshape(1, -1),
                neighborhood(X[i].reshape(1, -1), X,
                    factor=0.001, size=5),
            ) for i in range(10)])

            assert all(jaccards >= 0.0)
            assert all(jaccards <= 1.0)
            assert all(np.isfinite(jaccards))

    
@pytest.mark.parametrize("explainer,ds_name", zip(
    explainers.__all__,
    ds_names[:len(explainers.__all__)]
))
def test_infidelity(explainer, ds_name):
    # We'll explain the original equation and check if the stability value
    # is within the expected range, and how it deals with nans.

    X, y = load_samples_by_ds_name(ds_name)    

    explainer_instance = getattr(explainers, explainer)(
        predictor=Feynman_regressor(equation_name=ds_name).fit(X, y)
    )
    
    # Testing only for applicable explainers
    if (explainer_instance.agnostic == True or \
        'Feynman_regressor' in explainer_instance.agnostic) and \
        (explainer_instance.local_scope == True):
            
            explainer_instance.fit(X, y)

            # We'll explain 10 instances and see if all passes on the tests
            infidelities = np.array([infidelity(
                explainer_instance.explain_local,
                explainer_instance.predictor,
                X[i].reshape(1, -1),
                neighborhood(X[i].reshape(1, -1), X,
                    factor=0.001, size=5),
            ) for i in range(10)])

            assert all(infidelities >= 0.0)
            assert all(np.isfinite(infidelities))