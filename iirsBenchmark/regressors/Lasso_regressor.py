# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 21-11-2021 by Guilherme Aldeia

"""
Linear regressor with L1 regularization (Lasso). This method is considered a
white-box for most authors, and it is also used in some cases for feature
selection, given that the regularization leads to possibly sparse coefficients,
where many features are discarded.

This regressor extends the scikit-learn
[Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).

beyond what the scikit model can do, this class also implements:
* `to_str()` method, that returns a string representation of the final model;
* `gradients()` method, which computes the gradient vector for a given
    observation (or observations);
* `stochastic_executions` attribute, indicating if the model presents
    different results between different executions if the random_state is not
    setted;
* `interpretability_spectrum` attribute, with a string representing where on
    the interpretability spectrun (white, gray or black-box) this model lays;
* `grid_params` attribute, with different possible values to be used in 
    a gridsearch optimization of the method;
* `feature_importances_` attribute, representing the importances calculated by
    an intrinsic explanation method (the Partial Effect, used in the context
    of regression analysis).
"""


from sklearn.linear_model import Lasso

import numpy as np


class Lasso_regressor(Lasso):
    def __init__(self, *, 
        alpha=1.0, fit_intercept=True, normalize=False, precompute=False,
        copy_X=True, max_iter=1000, tol=0.0001, warm_start=False,
        positive=False, selection='cyclic', random_state=None, **kwargs):
        
        # if selection='cyclic, then random_state doesn't make any difference

        super(Lasso_regressor, self).__init__(
            alpha=alpha, fit_intercept=fit_intercept, normalize=normalize,
            precompute=precompute, copy_X=copy_X, max_iter=max_iter,
            tol=tol, warm_start=warm_start, positive=positive,
            random_state=random_state, selection=selection)
        

    def fit(self, X, y):
        super_fit =  super().fit(X, y)

        self.feature_importances_ = self.coef_

        n_features = X.shape[1]

        # Useful for model specific explainers that uses information about
        # selected features
        self.selected_features_ = np.array(
            [i for i in range(n_features) if self.coef_[i] != 0.0])

        return super_fit


    def to_str(self):
        coefs     = self.coef_
        intercept = self.intercept_
            
        str_terms = []
        for i, c in enumerate(coefs):
            if np.isclose(c, 0.0):
                continue
            
            str_terms.append(f"{c.round(3)}*x_{i}")

        expr_str = ' + '.join(str_terms)

        return f"{expr_str} + {intercept.round(3)}"


    def gradients(self, X):

        gradients = np.zeros_like(X)

        # broadcasting coefficients, since the gradients will always
        # be the same.
        gradients[:] = self.coef_

        return gradients
        

Lasso_regressor.interpretability_spectrum = 'white-box'
Lasso_regressor.stochastic_executions = False
Lasso_regressor.grid_params = {
    'alpha' : [0.001, 0.01, 0.1, 1, 10],
}