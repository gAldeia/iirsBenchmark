# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Interpretability in Regression: a Benchmark of Explanatory methods (iirBenchmark)

this library is intended to unify the creation and use of the regressors and
explainers that make up the repository's experiments.

The experiments seek to quantitatively assess the performance of different
exponents of regression models, using various metrics to compare global and
local explanations. For this, a set of 100 physics equations was used to
generate the data used. The experiments were carried out on the
[Feynman database](https://space.mit.edu/home/tegmark/aifeynman.html).

There are 3 main sub-modules:
* `feynman`: implements a regressor that takes as argument the name of the 
    Feynman equation label and behaves like the original
    physical equation used to create the data. This will be the best
    estimator as it has complete internal knowledge of the data used.
    It is expected that the explanations and metrics calculated on this
    regressor can be used to baseline the other regressors;
* `regressors`: implementation of the regressors so that they are easy to use in
    the experiment. In general, the implementations here are just
    extensions of the original implementations, since there is
    already a convention in python for the development of these
    types of methods, through the inheritance of the RegressorMixin
    class from the learn scikit;
* `explainers`: module that implements, with a similar interface, several
    state-of-the-art explainers in the literature and widely used.
    Unlike supervised machine learning methods for regression, there is no
    convention for the structure of explainers (something like scikit's
    RegressorMixin). The intention of this sub-module is precisely
    to make this unification.
"""

import jax

jax.config.update('jax_platform_name', 'cpu')
