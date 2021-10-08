
# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
k-Nearest Neighbors regressor. This method was already reported in the
literature as white and black box. This package will consider kNN as a
gray-box model. It can be interpretable in the sense that it is a
template matching algorithm (and prototypes are known as one type of
explanation), but it does not provide feature importances nor any 
concrete equation to be inspected (being not interpretable from this
perspective). Since all explainers in this package focus on feature
importance attribution, kNN will lay on the middle of the spectrum.

This regressor extends the scikit-learn
[KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html).

beyond what the scikit model can do, this class also implements:
* `to_str()` method, that returns a string representation of the final model;
* `stochastic_executions` attribute, indicating if the model presents
    different results between different executions if the random_state is not
    setted;
* `interpretability_spectrum` attribute, with a string representing where on
    the interpretability spectrun (white, gray or black-box) this model lays;
* `grid_params` attribute, with different possible values to be used in 
    a gridsearch optimization of the method.
"""

from sklearn.neighbors import KNeighborsRegressor


class KNN_regressor(KNeighborsRegressor):
    def __init__(self, *,
        n_neighbors=5, weights='uniform', algorithm='auto',
        leaf_size=30, p=2, metric='minkowski', metric_params=None, **kwargs):

        # This method does not have a stochastic behavior and the scikit
        # implementation does not take a random_state argument

        # the scikit method supports parallelization, but we want to avoid 
        # nested paralellizations (the original experiments were designed
        # to run in multiple subprocesses). n_jobs of superclass should be None

        super(KNN_regressor, self).__init__(
            n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
            leaf_size=leaf_size, p=p, metric=metric,
            metric_params=metric_params,
            n_jobs=None)
        

    def to_str(self):
        return ("scikit kNN black-box model with params: "
               f"{str(self.get_params())}")


KNN_regressor.stochastic_executions = False
KNN_regressor.interpretability_spectrum = 'gray-box'
KNN_regressor.grid_params = {
    'n_neighbors' : [3, 5, 7, 9, 11, 17, 19, 23, 29, 31],
}