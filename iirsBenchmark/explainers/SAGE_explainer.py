# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.1.0
# Last modified: 12-29-2021 by Guilherme Aldeia

"""
SAGE explainer
"""

from iirsBenchmark.explainers._base_explainer import Base_explainer 

from sklearn.utils import check_X_y

import sage #https://github.com/iancovert/sage

import numpy as np


class SAGE_explainer(Base_explainer):
    def __init__(self, *,
        predictor, n_permutations=1e10, n_samples=1024, **kwargs):

        super(SAGE_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = False,
            global_scope = True
        )

        self.n_permutations = n_permutations
        self.n_samples      = n_samples
        
        
    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y

        # using fewer samples
        if self.n_samples != None:
            self.X_indexes_ = np.random.choice(
                X.shape[0], size=min(self.n_samples, X.shape[0]), replace=False)
        else:
            self.X_indexes_ = np.arange(X.shape[0])

        self.X_samples_ = X[self.X_indexes_, :]

        
        self.imputer   = sage.MarginalImputer(self.predictor, self.X_samples_)
        self.estimator = sage.PermutationEstimator(self.imputer, 'mse')

        return self

        
    def explain_global(self, X, y):
        
        self._check_is_fitted()

        X, y = check_X_y(X, y)

        assert X.shape[0] > 1, \
            "Global explanation should have at least two observations." 

        sage_values = self.estimator(X, y, verbose=False, bar=False,
                                n_permutations=self.n_permutations)
                                
        explanations = np.abs(sage_values.values).reshape(1, -1)
        
        return self._protect_explanation(explanations)