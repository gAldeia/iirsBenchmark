# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Random feature importance attribution explainer
"""

from iirsBenchmark.explainers._base_explainer import Base_explainer 

from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array

import numpy as np


class RandomImportance_explainer(Base_explainer):
    def __init__(self, *, predictor, **kwargs):
        super(RandomImportance_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = True,
            global_scope = True
        )
        

    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y


    def explain_local(self, X):
        
        self._check_is_fitted()

        X = check_array(X)

        explanations = np.zeros_like(X)

        for i in range(X.shape[0]):

            # we add one to have non-zero importances
            explanations[i, :] = np.random.permutation(X.shape[1]) + 1

        return self._protect_explanation(explanations)


    def explain_global(self, X, y):
        
        self._check_is_fitted()

        X, y = check_X_y(X, y)

        assert X.shape[0] > 1, \
            "Global explanation should have at least two observations." 

        explanation = (np.random.permutation(X.shape[1]) + 1).reshape(1, -1)

        return self._protect_explanation(explanation) 