# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
LIME explainer
"""

import lime
import lime.lime_tabular

from iirsBenchmark.explainers._base_explainer import Base_explainer 

from sklearn.utils.validation import check_array

import numpy as np


class LIME_explainer(Base_explainer):
    def __init__(self, *, predictor, num_samples=30, **kwargs):
        super(LIME_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = True,
            global_scope = False
        )

        self.num_samples = num_samples
        
    
    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y
        
        self._LIME_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_, verbose=False, mode='regression')

        return self


    def explain_local(self, X):
        
        self._check_is_fitted()

        X = check_array(X)
        
        nobs, nvars = X.shape

        explanations = np.zeros_like( X )
        for i in range(nobs):
            explanation = self._LIME_explainer.explain_instance(
                X[i], self.predictor.predict, num_features=nvars,
                num_samples=self.num_samples)

            for (feature_id, weight) in explanation.as_map()[0]:
                explanations[i, feature_id] = weight

        return self._protect_explanation(explanations)