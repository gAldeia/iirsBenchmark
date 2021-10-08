# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Morris sensitivity explainer
"""

from iirsBenchmark.explainers._base_explainer import Base_explainer 

from sklearn.utils import check_X_y

from interpret.blackbox import MorrisSensitivity
from interpret.blackbox.sensitivity import MorrisSampler #versão 0.2.5


class MorrisSensitivity_explainer(Base_explainer):
    def __init__(self, *, predictor, **kwargs):
        super(MorrisSensitivity_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = False,
            global_scope = True
        )


    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y
        
        
    def explain_global(self, X, y):
        
        self._check_is_fitted()

        X, y = check_X_y(X, y)

        assert X.shape[0] > 1, \
            "Global explanation should have at least two observations." 

        sampler = MorrisSampler(
            data = self.X_,
            feature_names = [f'x_{i}' for i in range(self.X_.shape[1])],
            N = 100,
        )

        _msa = MorrisSensitivity(
            predict_fn=self.predictor.predict,
            data=self.X_, 
            feature_types = ['numerical' for _ in range(self.X_.shape[1])]
            # sampler=sampler
        )

        msa_global = _msa.explain_global()
 
        explanation = msa_global.data()['scores'].reshape(1, -1)
        
        return self._protect_explanation(explanation)