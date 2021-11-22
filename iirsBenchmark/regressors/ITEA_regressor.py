
# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 21-11-2021 by Guilherme Aldeia

"""
Symbolic regressor with the IT representation. This method is considered by 
the authors as a gray-box method.

This regressor extends the
[ITEA_regressor](https://galdeia.github.io/itea-python/itea.regression.html),
which is a subclass of scikits RegressorMixin.

beyond what the model can do, this class also implements:
* `to_str()` method, that returns a string representation of the final model;
* `gradients()` method, which computes the gradient vector for a given
    observation (or observations);
* `stochastic_executions` attribute, indicating if the model presents
    different results between different executions if the random_state is not
    setted;
* `interpretability_spectrum` attribute, with a string representing where on
    the interpretability spectrun (white, gray or black-box) this model lays;
* `grid_params` attribute, with different possible values to be used in 
    a gridsearch optimization of the method;
* `feature_importances_` attribute, representing the importances calculated by
    an intrinsic explanation method (the Partial Effect, used in the context
    of regression analysis).
* `selected_features_`: some model-specific explainers can rely on using a
    subset of the original features to explain the model (i.g. ELA). This
    attribute is created after fitting the regressor, and indicates the indexes
    of the features that are actually being considered by the model when making
    predictions, and it is implemented to all regressors that creates a 
    mathematical expression that can be analysed to extract this information.
"""

import itea.regression
from itea.inspection import ITExpr_explainer

# jax version 0.2.13
from jax import grad, vmap
import jax.numpy as jnp 

from sklearn.metrics import mean_squared_error, r2_score

# Currently (08-20-2021) the ITEA does not support changing the evaluation
# metric for the training (and uses RMSE). Below is a workaround to change it
# to MSE (or R2 if needed)

class ITExpr_regressor(itea.regression.ITExpr_regressor):
    def __init__(self, *, expr, tfuncs, labels = [], **kwargs):

        super(ITExpr_regressor, self).__init__(
            expr   = expr,
            tfuncs = tfuncs,
            labels = labels,
            **kwargs
        )

        # Here we can set the metric for the fitness function
        self.fitness_f = lambda pred, y: mean_squared_error(
            pred, y, squared=True)


# ITEA_regressor searches for the best ITExpr_regressor
class ITEA_regressor(itea.regression.ITEA_regressor):
    def __init__(self, *, 
        gens=100, popsize=100, expolim=(-4, 4), max_terms=10,
        tfuncs={
            'log'   : jnp.log,
            'sqrt'  : lambda x: jnp.sqrt(x),
            'id'    : lambda x: x,
            'sin'   : jnp.sin,
            'cos'   : jnp.cos,
            'tanh'  : jnp.tanh,
            'exp'   : jnp.exp,
            'expn'  : lambda x: jnp.exp(-x),
            'arcsin': jnp.arcsin
        },
        simplify_method=None, random_state=None, verbose=None,
        labels=[], **kwargs):

        # derivatives will be generated using jax autodif
        tfuncs_dx = dict()

        for k, v in tfuncs.items():
            tfuncs_dx[k] = vmap(grad(v))

        super(ITEA_regressor, self).__init__(
            gens=gens, popsize=popsize, tfuncs=tfuncs, tfuncs_dx=tfuncs_dx,
            expolim=expolim, max_terms=max_terms,
            simplify_method=simplify_method, random_state=random_state,
            verbose=verbose, labels=labels, **kwargs)

        self.itexpr_class = itea.regression.ITExpr_regressor

        # For ITEA we need to specify if a smaller (or greater) fitness is
        # better
        self.greater_is_better = False


    def to_str(self):
        return self.bestsol_.to_str()

    
    def gradients(self, X):

        return self.bestsol_.gradient(
            X, self.tfuncs_dx, logit=False)[0]
        

    def fit(self, X, y):
        super_fit =  super().fit(X, y)

        # Useful for model specific explainers that uses information about
        # selected features
        self.selected_features_ = ITExpr_explainer(
            itexpr=self.bestsol_, tfuncs=self.tfuncs).selected_features(idx=True)

        return super_fit


ITEA_regressor.grid_params = {
    'popsize' : [100, 250, 500],
    'gens'    : [100, 250, 500],
}
ITEA_regressor.interpretability_spectrum = 'gray-box'
ITEA_regressor.stochastic_executions = True