# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 21-11-2021 by Guilherme Aldeia

"""
Linear regressor. This method is considered a white-box for most authors.

This regressor extends the scikit-learn
[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

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

from sklearn.linear_model import LinearRegression

import numpy as np

class Linear_regressor(LinearRegression):
    def __init__(self, *,
        fit_intercept=True, normalize=False, copy_X=True,
        positive=False, **kwargs):

        # This method does not have a stochastic behavior and the scikit
        # implementation does not take a random_state argument
        
        # the scikit method supports parallelization, but we want to avoid 
        # nested paralellizations (the original experiments were designed
        # to run in multiple subprocesses). n_jobs of superclass should be None

        super(Linear_regressor, self).__init__(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
            n_jobs=None, positive=positive)
        

    def fit(self, X, y):
        super_fit =  super().fit(X, y)

        self.feature_importances_ = self.coef_

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


Linear_regressor.interpretability_spectrum = 'white-box'
Linear_regressor.stochastic_executions = False
Linear_regressor.grid_params = {

}