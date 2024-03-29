# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 21-11-2021 by Guilherme Aldeia

"""
Explain by Local Approximation explainer.
"""

from iirsBenchmark.explainers._base_explainer import Base_explainer 

from sklearn.utils.validation import check_array
from sklearn.linear_model import LinearRegression

import numpy as np


class ELA_explainer(Base_explainer):
    def __init__(self, *, predictor, k=5, **kwargs):
        
        super(ELA_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = ['ITEA_regressor', 'Operon_regressor', 'GPbenchmark_regressor',
                'Linear_regressor', 'Lasso_regressor', 'Feynman_regressor'],
            local_scope  = True,
            global_scope = False
        )
        
        # (global explanation is not a feature importance, but a visual
        # explanation like PartialDependencePlots, so it was not implemented
        # here)
        
        # n of closest neighbors evaluated in the linear regression
        self.k = k 

    
    def _check_fit(self, X, y):
         
        assert X.shape[0] >= self.k, \
            f"Data set too small to be used with given value for k={self.k}." 

        return super()._check_fit(X, y)


    def _k_closest_neighbors(self, x):
        
        # p1 and p2 must be a 1-dimensional numpy array of same length
        euclidean_dist = lambda p1, p2: np.sqrt(np.sum((p1 - p2)**2))
        
        # Distance will consider only the subset of features existing in the
        # regresison model (the model must have a selected_features_ thus
        # ELA is not entirelly model agnostic).
        subset_features = self.predictor.selected_features_

        # setting discarded variables to same value so it doesn't affect
        # the distance calculation
        x_masked = x.copy()
        X_masked = self.X_.copy()

        # x is 1d, X is 2d
        x_masked[subset_features] = 0.0
        X_masked[:, subset_features] = 0.0

        selected = np.argsort(
            [euclidean_dist(x_masked, xprime) for xprime in X_masked])
        
        return self.X_[selected[ :self.k], :]
        
        
    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y
        
        return self
        

    def explain_local(self, X):
        
        self._check_is_fitted()

        X = check_array(X)
        
        nobs, nvars = X.shape
        
        coeffs = np.zeros_like(X)
        for i in range(nobs):
            X_closest_neighbors = self._k_closest_neighbors(X[i, :])
            
            linear_reg = LinearRegression()
            linear_reg.fit(
                X_closest_neighbors, self.predictor.predict(X_closest_neighbors))
            
            coeffs[i, :] = linear_reg.coef_
            
        # Final explanation is the product of x by the coefficients
        # normalized for all variables
        explanations = np.abs(coeffs * X)
        
        # Normalizing (broadcast and transposing to divide matrix by vector)
        explanations = ((explanations * 100.).T / np.sum(explanations, axis=1)).T
        
        # check if everything is as expected: column-wise sum
        # should be 100
        assert np.all(np.isclose(np.sum(explanations, axis=1), 100.0))
        
        return self._protect_explanation(explanations)