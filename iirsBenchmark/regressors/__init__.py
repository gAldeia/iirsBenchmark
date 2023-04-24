# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Implementation of a class derived from the regressors (which already follow
the scikit pattern). The implemented class is an extension used to add some
methods to unify the calling and use of regressors, and additional attributes
that can be useful for comparing regressors.

The implementations are compatible with scikit's GridsearchCV, implement the
`fit` and `predict` functions, and have a `score` method that returns the
R2 metric to be used in gridsearch.

The `__all__` attribute of this sub-module can be accessed to easily iterate
through the regressors.

The naming pattern used was `<CapitalizedCamelCaseRegressorName>_regressor.

The scikit version used (that provides most of the regressors) was 0.24.2.

Every method that has a stochastic behavior can have its execution fixed by
setting `random_state` to a integer value. This is useful to create an
(almost) fully replicable experiment.

Every constructor argument shoud be passed as a named argument, and all
regressors supports **kwargs, which means that a universal dictionary of
parameters can be passed to all regressors regardless of the parameters
exists or not on the constructor signature.
"""

from iirsBenchmark.regressors.ITEA_regressor         import ITEA_regressor
from iirsBenchmark.regressors.MLP_regressor          import MLP_regressor
from iirsBenchmark.regressors.DecisionTree_regressor import DecisionTree_regressor
from iirsBenchmark.regressors.Linear_regressor       import Linear_regressor
from iirsBenchmark.regressors.Lasso_regressor        import Lasso_regressor
from iirsBenchmark.regressors.KNN_regressor          import KNN_regressor
from iirsBenchmark.regressors.SVM_regressor          import SVM_regressor
from iirsBenchmark.regressors.RF_regressor           import RF_regressor
from iirsBenchmark.regressors.XGB_regressor          import XGB_regressor
from iirsBenchmark.regressors.Operon_regressor       import Operon_regressor

import warnings

# ITEA warns when label argument is not provided. 
warnings.filterwarnings(action='ignore', module=r'itea')



__all__ = [
    'ITEA_regressor',
    'MLP_regressor',
    'Linear_regressor',
    'KNN_regressor', 
    'SVM_regressor',
    'RF_regressor',
    'XGB_regressor',
    'Lasso_regressor',
    'DecisionTree_regressor',
    'Operon_regressor'
]
