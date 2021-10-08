
# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Random Forest regressor. This bagging method is considered a
black-box for most authors.

This regressor extends the scikit-learn
[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

beyond what the scikit model can do, this class also implements:
* `to_str()` method, that returns a string representation of the final model;
* `stochastic_executions` attribute, indicating if the model presents
    different results between different executions if the random_state is not
    setted;
* `interpretability_spectrum` attribute, with a string representing where on
    the interpretability spectrun (white, gray or black-box) this model lays;
* `grid_params` attribute, with different possible values to be used in 
    a gridsearch optimization of the method.
* `feature_importances_` attribute, representing the importances calculated by
    an intrinsic explanation method (the Partial Effect, used in the context
    of regression analysis).
"""

from sklearn.ensemble import RandomForestRegressor


class RF_regressor(RandomForestRegressor):
    def __init__(self, *, 
        n_estimators=100, criterion='mse', max_depth=None,
        min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features='auto',
        max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, bootstrap=True,
        oob_score=False, verbose=0, warm_start=False,
        ccp_alpha=0.0, max_samples=None, random_state=None, **kwargs):

        # the scikit method supports parallelization, but we want to avoid 
        # nested paralellizations (the original experiments were designed
        # to run in multiple subprocesses). n_jobs of superclass should be None

        super(RF_regressor, self).__init__(
            n_estimators=n_estimators, criterion=criterion,
            max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features, max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, bootstrap=bootstrap,
            oob_score=oob_score, n_jobs=None, random_state=random_state,
            verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha,
            max_samples=max_samples)
        
        
    def to_str(self):
        return ("scikit RF black-box model with params: "
               f"{str(self.get_params())}")


RF_regressor.stochastic_executions = True
RF_regressor.interpretability_spectrum = 'black-box'
RF_regressor.grid_params = {
    'n_estimators'      : [100, 200, 300],
    'min_samples_split' : [0.01, 0.05, 0.1],
}