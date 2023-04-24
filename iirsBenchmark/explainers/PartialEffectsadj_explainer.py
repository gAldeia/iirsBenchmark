# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Partial Effects adjusted to retrieve shapley values explainer.
"""

from iirsBenchmark.explainers._base_explainer          import Base_explainer 
from iirsBenchmark.explainers.PartialEffects_explainer import PartialEffects_explainer 

from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array

import numpy as np


class PartialEffectsadj_explainer(Base_explainer):
    def __init__(self, *, predictor, **kwargs):
        super(PartialEffectsadj_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = ['ITEA_regressor', 'Operon_regressor', 'GPbenchmark_regressor',
                'Linear_regressor', 'Lasso_regressor', 'Feynman_regressor'],
            local_scope  = True,
            global_scope = True
        )
        
        self._partialEffects = PartialEffects_explainer(predictor=predictor)
        
    
    def _adjust(self, gradients, X):
        
        explanations = np.zeros_like( X )
        for i in range(X.shape[1]):
            explanations[:, i] = \
                gradients[:, i].mean() * (X[:, i] - X[:, i].mean())
            
        return explanations


    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y
        
        self._partialEffects.fit(X, y)

        return self
        

    def explain_local(self, X):
        
        self._check_is_fitted()

        X = check_array(X)
        
        explanations = self._adjust(
            self._partialEffects.explain_local(X),
            X
        )

        return self._protect_explanation(explanations)


    def explain_global(self, X, y):
        
        self._check_is_fitted()

        X, y = check_X_y(X, y)

        assert X.shape[0] > 1, \
            "Global explanation should have at least two observations." 
        
        explanation = np.mean(np.abs(self._adjust(
            self._partialEffects.explain_local(X),
            X)), axis=0).reshape(1, -1)

        return self._protect_explanation(explanation)