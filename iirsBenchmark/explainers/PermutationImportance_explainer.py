# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Feature permutation explainer
"""

from iirsBenchmark.explainers._base_explainer import Base_explainer 

from sklearn.utils import check_X_y

from sklearn.inspection import permutation_importance


class PermutationImportance_explainer(Base_explainer):
    def __init__(self, *, predictor, n_repeats=30, **kwargs):

        super(PermutationImportance_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = False,
            global_scope = True
        )

        self.n_repeats = n_repeats


    def fit(self, X, y):

        self._check_fit(X, y)
        self.X_ = X
        self.y_ = y

        return self


    def explain_global(self, X, y):

        self._check_is_fitted()

        X, y = check_X_y(X, y)

        assert X.shape[0] > 1, \
            "Global explanation should have at least two observations." 

        explanation = permutation_importance(
            self.predictor, X, y,
            n_repeats=self.n_repeats,
            random_state=0
        ).importances_mean.reshape(1, -1)

        return self._protect_explanation(explanation)