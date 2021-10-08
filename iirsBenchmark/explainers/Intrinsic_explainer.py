# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Intrinsic explainer. Some regressors have intrinsic methods to evaluate
a global feature importance, that can be accessed by `feature_importances_`
attribute. This explainer knows which of the iirBenchmark.regressors have
this feature importance and works with them.
"""

from iirsBenchmark.explainers._base_explainer import Base_explainer 
from sklearn.utils import check_X_y


class Intrinsic_explainer(Base_explainer):
    def __init__(self, *, predictor, **kwargs):
        super(Intrinsic_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = [
                'ITEA_regressor', 'DecisionTree_regressor', 
                'Linear_regressor', 'Lasso_regressor', 
                'RF_regressor', 'XGB_regressor'],
            local_scope  = False,
            global_scope = True
        )


    def fit(self, X, y):
        self._check_fit(X, y)

        # The data doesn't matter for this method
        self.fitted_ = True
        

    def explain_global(self, X, y):
        
        self._check_is_fitted()
        
        X, y = check_X_y(X, y)

        explanation = self.predictor.feature_importances_
        
        if explanation.ndim == 1:
            return explanation.reshape(1, -1)
        
        return self._protect_explanation(explanation)