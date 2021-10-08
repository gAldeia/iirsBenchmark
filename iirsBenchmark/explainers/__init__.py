# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Implementation of classes to provide a unified way of using state-of-the-art
explanatory methods in the literature and widely used.
    
Unlike supervised machine learning methods for regression, there is no 
convention for the structure of explainers (something like the scikit's
RegressorMixin). The intention of this sub-module is precisely to make this
unification.

Explainers can be model-agnostic or dependent, and they can support local or
global explanations (or both). In case you try to create an instance of an
explainer that is not applicable, a `NotApplicableException` error will be
thrown, and its used to allow the user to catch and handle those cases.

The `__all__` attribute of this sub-module can be accessed to easily iterate
through the explainers.

The naming pattern used was `<CapitalizedCamelCaseRegressorName>_explainer.
"""

from iirsBenchmark.explainers.PermutationImportance_explainer import PermutationImportance_explainer
from iirsBenchmark.explainers.RandomImportance_explainer      import RandomImportance_explainer
from iirsBenchmark.explainers.SHAP_explainer                  import SHAP_explainer
from iirsBenchmark.explainers.SHAPadj_explainer               import SHAPadj_explainer
from iirsBenchmark.explainers.SAGE_explainer                  import SAGE_explainer
from iirsBenchmark.explainers.LIME_explainer                  import LIME_explainer
from iirsBenchmark.explainers.PartialEffects_explainer        import PartialEffects_explainer
from iirsBenchmark.explainers.PartialEffectsadj_explainer     import PartialEffectsadj_explainer
from iirsBenchmark.explainers.IntegratedGradients_explainer   import IntegratedGradients_explainer
from iirsBenchmark.explainers.MorrisSensitivity_explainer     import MorrisSensitivity_explainer
from iirsBenchmark.explainers.Intrinsic_explainer             import Intrinsic_explainer

__all__ = [
    'PermutationImportance_explainer',
    'RandomImportance_explainer',
    'SHAP_explainer',
    'SHAPadj_explainer',
    'SAGE_explainer',
    'LIME_explainer',
    'IntegratedGradients_explainer',
    'Intrinsic_explainer',
    'PartialEffectsadj_explainer',
    'PartialEffects_explainer',
    'MorrisSensitivity_explainer',
]