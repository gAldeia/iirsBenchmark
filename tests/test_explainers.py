import pytest

import pandas as pd

import iirsBenchmark.explainers as explainers

from iirsBenchmark.feynman    import Feynman_regressor
from iirsBenchmark.exceptions import NotApplicableException


ds_names = pd.read_csv(
    './datasets/FeynmanEquations.csv')['Filename'].values


@pytest.mark.parametrize("explainer", explainers.__all__)
def test_has_properties_and_methods(explainer):
    
    # checking attributes does not need to instantiate the class
    explainer_class = getattr(explainers, explainer)
    explainer_instance = explainer_class(
        predictor=Feynman_regressor(equation_name='I.10.7'))

    assert hasattr(explainer_instance, 'predictor')
    assert hasattr(explainer_instance, 'agnostic')
    assert hasattr(explainer_instance, 'local_scope')
    assert hasattr(explainer_instance, 'global_scope')

    # methods used in the experiments. All explainers should have those
    for method in ['fit', 'explain_local', 'explain_global']:
        # should throw exception if the class does not have to_string
        hasattr(explainer_instance, method)

        assert callable(getattr(explainer_instance, method))

    
@pytest.mark.parametrize("explainer,ds_name", zip(
    explainers.__all__,
    ds_names[:len(explainers.__all__)]
))
def test_fit_and_explain_local(explainer, ds_name):

    # testing with random datasets. The dataset correctness should be
    # verified in test_feynman.
    data = pd.read_csv(
        f'./datasets/test/{ds_name}_LHS.csv', sep=',', 
        header=0, index_col=False).values

    X, y = data[:, :-1], data[:, -1]

    explainer_class = getattr(explainers, explainer)
    explainer_instance = explainer_class(
        predictor=Feynman_regressor(equation_name=ds_name).fit(X, y)
    )
    
    if explainer_instance.agnostic == True or \
         'Feynman_regressor' in explainer_instance.agnostic:

        explainer_instance.fit(X, y)

        if explainer_instance.local_scope == True:
            explainer_instance.explain_local(X[:2, :])
        else:
            # here it should fail
            with pytest.raises(NotApplicableException):
                explainer_instance.explain_local(X[:2, :])
    else:
        # here it should fail (intrinsic explainer)
        with pytest.raises(NotApplicableException):
            explainer_instance.fit(X, y)
            

@pytest.mark.parametrize("explainer,ds_name", zip(
    explainers.__all__,
    ds_names[:len(explainers.__all__)]
))
def test_fit_and_explain_global(explainer, ds_name):

    # testing with random datasets. The dataset correctness should be
    # verified in test_feynman.
    data = pd.read_csv(
        f'./datasets/test/{ds_name}_LHS.csv', sep=',', 
        header=0, index_col=False).values

    X, y = data[:, :-1], data[:, -1]

    explainer_class = getattr(explainers, explainer)
    explainer_instance = explainer_class(
        predictor=Feynman_regressor(equation_name=ds_name).fit(X, y)
    )
    
    if explainer_instance.agnostic == True or \
        'Feynman_regressor' in explainer_instance.agnostic:

        explainer_instance.fit(X, y)

        if explainer_instance.global_scope == True:
            explainer_instance.explain_global(X[:2, :], y[:2])
        else:
            # here it should fail
            with pytest.raises(NotApplicableException):
                explainer_instance.explain_global(X[:2, :], y[:2])
    else:
        # here it should fail (intrinsic explainer)
        with pytest.raises(NotApplicableException):
            explainer_instance.fit(X, y)