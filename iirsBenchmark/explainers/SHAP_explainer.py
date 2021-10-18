# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Shapley Values explainer
"""

import shap #0.34.0

from iirsBenchmark.explainers._base_explainer import Base_explainer 

from sklearn.utils import check_X_y
from sklearn.utils.validation import check_array

import numpy as np


class SHAP_explainer(Base_explainer):
    def __init__(self, *,
        predictor, l1_reg='num_features(10)', sample_size=30, **kwargs):
        
        super(SHAP_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = True,
            global_scope = True
        )
        
        self.l1_reg = l1_reg
        self.sample_size = sample_size


    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y
        
        if (np.issubdtype(type(self.sample_size), int) and
            self.sample_size != -1):

            sampled_data = shap.sample(
                self.X_, np.minimum(self.X_.shape[0], self.sample_size))
        else:
            sampled_data = self.X_

        self._SHAP_explainer = shap.KernelExplainer(
            self.predictor.predict,
            sampled_data,
            silent=True)

        return self


    def explain_local(self, X):
        
        self._check_is_fitted()

        X = check_array(X)

        explanations = self._SHAP_explainer.shap_values(
            X, l1_reg=self.l1_reg, silent=True)

        return self._protect_explanation(explanations)


    def explain_global(self, X, y):
        
        self._check_is_fitted()

        X, y = check_X_y(X, y)

        assert X.shape[0] > 1, \
            "Global explanation should have at least two observations." 

        shapley_values = self._SHAP_explainer.shap_values(
            X, l1_reg='num_features(10)', silent=True)
        
        explanation = np.mean(np.abs(shapley_values), axis=0).reshape(1, -1)

        return self._protect_explanation(explanation)