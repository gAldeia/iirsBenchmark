# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Extreme Gradient Boosting regressor. This boosting method is considered a
black-box for most authors.

This regressor extends the scikit-learn
[GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor).

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

from sklearn.ensemble import GradientBoostingRegressor


class XGB_regressor(GradientBoostingRegressor):
    def __init__(self, *, 
        loss='ls', learning_rate=0.1, n_estimators=100,
        subsample=1.0, criterion='friedman_mse',
        min_samples_split=0.5, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_depth=3,
        min_impurity_decrease=0.0, min_impurity_split=None,
        init=None, random_state=None, max_features=None,
        alpha=0.9, verbose=0, max_leaf_nodes=None,
        warm_start=False, validation_fraction=0.1,
        n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0, 
        **kwargs):

        super(XGB_regressor, self).__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            subsample=subsample, criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split, init=init,
            random_state=random_state, max_features=max_features,
            alpha=alpha, verbose=verbose, max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)
        

    def to_str(self):
        return ("scikit XGB black-box model with params: "
               f"{str(self.get_params())}")


XGB_regressor.interpretability_spectrum = 'black-box'
XGB_regressor.stochastic_executions = True
XGB_regressor.grid_params = {
    'n_estimators' : [100, 200, 300],
    'min_samples_split' : [0.01, 0.05, 0.1],
}