# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Shapley Values adjusted to retrieve IG values explainer
"""

import shap #0.34.0

from iirsBenchmark.explainers._base_explainer import Base_explainer  
from iirsBenchmark.explainers.SHAP_explainer import SHAP_explainer 

from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array

import numpy as np

class SHAPadj_explainer(Base_explainer):
    def __init__(self, *,
        predictor, l1_reg='num_features(10)', sample_size=30, **kwargs):
        
        super(SHAPadj_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = True,
            global_scope = True
        )
                
        self.l1_reg = l1_reg
        self.sample_size = sample_size

        self._shap = SHAP_explainer(
            predictor=predictor, l1_reg = l1_reg, sample_size=sample_size)


    def _signal(self, x):
        z = np.sign(x)
        z[z==0] =  1

        return z
    
    
    def _adjust(self, shapley_values, X):
        
        explanations = np.zeros_like(shapley_values)

        # Adjusting to make SHAP approximate PE/IG
        for i in range(X.shape[1]): # Para cada variável

            # sign will be used to prevent loosing sign information
            mean_diff = (X[:, i] - self.X_[:, i].mean())
            explanations[:, i] = (
                (self._signal(mean_diff) * shapley_values[:, i]) /
                np.sqrt(1 + np.power(mean_diff, 2))
            )

            # using the analytic quotient to avoid division by numbers close
            # to zero
            
        return explanations


    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y
        
        self._shap.fit(X, y)

        return self


    def explain_local(self, X):
        
        self._check_is_fitted()

        X = check_array(X)

        explanations = self._adjust(self._shap.explain_local(X), X)
            
        return self._protect_explanation(explanations)


    def explain_global(self, X, y):
        
        self._check_is_fitted()

        X, y = check_X_y(X, y)

        assert X.shape[0] > 1, \
            "Global explanation should have at least two observations." 

        # Final explanation is the mean of absolute explanations
        explanation = np.mean(np.abs(self._adjust(
            self._shap.explain_local(X), X)), axis=0).reshape(1, -1)

        return self._protect_explanation(explanation)