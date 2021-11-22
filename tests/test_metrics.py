import pytest

import pandas as pd
import numpy as np

import iirsBenchmark.explainers as explainers
from iirsBenchmark.metrics import (
    stability, jaccard_stability, infidelity, neighborhood)

from iirsBenchmark.feynman    import Feynman_regressor


ds_names = pd.read_csv(
    './datasets/FeynmanEquations.csv')['Filename'].values


def load_samples_by_ds_name(ds_name):
    data = pd.read_csv(
        f'./datasets/test/{ds_name}_LHS.csv', sep=',', 
        header=0, index_col=False).values

    # Using only the first 100 observations of the data
    X, y = data[:100, :-1], data[:100, -1]

    return X, y


@pytest.mark.parametrize("metric,bounds",[
    [stability, (0.0, np.inf)],
    [infidelity, (0.0, np.inf)],
    [jaccard_stability, (0.0, 1.0)],
    
])
@pytest.mark.parametrize("explainer,ds_name", zip(
    explainers.__all__,
    ds_names[:len(explainers.__all__)]
))
def test_metric(metric, bounds, explainer, ds_name):
    # We'll explain the original equation and check if the metric value
    # is within the expected range, and how it deals with nans.

    lower_b, upper_b = bounds

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
            metric_values = np.array([metric(
                explainer_instance,
                X[i].reshape(1, -1),
                neighborhood(X[i].reshape(1, -1), X,
                    factor=0.001, size=5),
            ) for i in range(10)])

            assert all(metric_values >= lower_b)
            assert all(metric_values <= upper_b)
            assert all(np.isfinite(metric_values))