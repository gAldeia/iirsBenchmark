# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y
import numpy as np

"""
Base explainer, should be inherited by all explainers.

The base explainer has the following attributes:
* `agnostic` : attribute indicating if the explainer is model agnostic or not.
    If it is, then agnostic=True, else agnostic is a list of strings containing
    the name of compatible regressors;
* `local_scope` : boolean indicating if the explainer supports local
    explanations. A local explanation is a feature importance attribution for
    a single observation, and does not depend to the y value;
* `global_scope` : boolean indicating if the explainer supports global
    explanations. A global explanation summarises the model feature importances
    by creating a single feature importance array. The global explanation
    can use training data, or can work independently of the data. In case
    the global explanation depends of the data used to explain, different 
    global explanations can be obtained, and when it is independent of the data
    the same explanation will always be returned.

And the following methods, that should be overwriten in subclasses:
* `_check_fit(self, X, y)` : checks if the explainer is compatible with the
    given predictor on the constructor, and if X and y are compatible and 
    correct arrays. Raises different exceptions if some check fails;
* `_protect_explanation(self, X)`: takes an explanation and protects the
    returned values if some invalid value is found. Can be overwritten for
    more specific cases. Internal use only, should take a matrix of explanations
    and will return a new matrix of same shape (n_observations, n_features)
* `_check_is_fitted(self)` : check if the explainer is fitted (when fitting
    an explainer, some attributes ending with an underscore are created 
    within the class instance) when calling `explain_local` or `explain_global`;
* `fit(X, y)`: fits the explainer with the training data;
* `explain_local(self, X)` : takes as argument a matrix of shape
    (n_observations, n_features) and return a local explanation for each
    observation;
* `explain_global(self, X, y)` : takes as argument a matrix with more than
    one sample and produces a global explanation, returning a matrix of shape
    (1, n_features).

To implement explainers, this class should be inherited and the
`_protect_explanation` and `fit`, should be overwriten. Methods 
`explain_local` and `explain_global` should be overwritten only if the
explainer supports those scope of explanations.

The constructor takes as argument, necessarly, a predictor instance
(not the predictor.predict!), and specifications about the subclass scope.
The constructor of the subclasses should take only the predictor and 
optional arguments with respect to the explainer itself. The `Base_explainer`
constructor is not meant to be used by the user.
"""


from iirsBenchmark.exceptions import NotApplicableException


class Base_explainer():
    def __init__(self, *, 
        predictor, agnostic=[], local_scope=False, global_scope=False):

        # By default, the explainer does not support any model or scope.

        self.predictor    = predictor
        
        self.agnostic     = agnostic
        self.local_scope  = local_scope
        self.global_scope = global_scope


    def _check_fit(self, X, y):
        X, y = check_X_y(X, y)

        # Checking if the given predictor is supported
        if (self.agnostic is not True and
            self.predictor.__class__.__name__ not in self.agnostic):

            raise NotApplicableException(
                f"The regressor {self.predictor.__class__.__name__} "
                "is not supported by this explainer.")


    def _protect_explanation(self, X):

        return np.ma.filled(np.ma.masked_outside(
            np.ma.masked_where(np.isnan(X) | np.isinf(X), X), -1e+10, 1e+10), 0)


    def _check_is_fitted(self):
        
        attrs = [v for v in vars(self)
                if v.endswith("_") and not v.startswith("__")]

        if not attrs:
            raise NotFittedError(
                "This explainers instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator.")
        

    def fit(self, X, y):
        
        raise NotImplementedError(
            f"This explainer {self.__class__.__name__} " 
            "is missing the implementation of the fit method!")


    def explain_local(self, X):
        
        raise NotApplicableException(
            f"The explainer {self.__class__.__name__} does not support "
            "local explanations.")
    

    def explain_global(self, X, y):

        raise NotApplicableException(
            f"The explainer {self.__class__.__name__} does not support "
            "local explanations.")