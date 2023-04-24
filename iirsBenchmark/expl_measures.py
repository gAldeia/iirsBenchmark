# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.2
# Last modified: 23-11-2021 by Guilherme Aldeia

"""
implementation of the metrics that will be used to assess the regressors
and explanations.
"""

import numpy as np

from sklearn.metrics          import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity


__all__ = [
    # Regression metrics
    'RMSE', 'R2',

    # Auxiliary methods for explanation metrics
    'neighborhood', 

    # Explanation robustness
    'stability', 'jaccard_stability', 'infidelity',

    # Explanation quality
    'cossim_expl', 'RMSE_expl'
]


# Neighborhood generation ------------------------------------------------------
def neighborhood(x, X_train, factor, size=100):
    """Method to create samples around a given observation x.

    This method uses a multivariate normal distribution to
    randomly select feature values. The sigma of the distribution
    is calculated over the training data to mimic the original
    distributions and a scaling factor is multiplied to
    adjust how large will be the neighborhood.

    It is possible to specify the number of generated samples
    by setting the size to a different value (default=100).

    Returns a matrix of shape (size, n_features) containing
    the sampled neighbors.
    
    """
    
    if x.ndim == 2:
        assert x.shape[0] == 1, \
            ("The neighborhood is created based only in a single observation. "
            f"The given data x has {x.shape[0]} observations.")

        # we need x to be a 1-dimensional array
        x = np.squeeze(x)

    if x.shape[0]==1:
        return np.random.normal(
            x, np.var(X_train)*factor, size=size).reshape(-1, 1)

    return np.random.multivariate_normal(
        x, np.cov(X_train.T)*factor, size=size)


# Regression metrics -----------------------------------------------------------
def RMSE(y, yhat):

    return mean_squared_error(y, yhat, squared=False)


def R2 (y, yhat):
    
    return r2_score(y, yhat)


# Quality metrics (related to groundtruth) -------------------------------------
def RMSE_expl(y, yhat):
    return mean_squared_error(y, yhat, squared=False)


def cossim_expl(y, yhat):
    
    return cosine_similarity(y, yhat)


# Robustness metrics and auxiliary methods -------------------------------------
def _norm_p2(vector):
    """p2 norm of a vector.

    the vector should be an array of shape (n_obs, n_samples).
    
    This method returns an array of shape (n_obs, 1) where each element
    is the norm calculated for the corresponding observation.
    """

    return np.sqrt(np.sum(
        np.abs(np.power(vector, 2)), axis=1
    )).reshape(-1, 1)


def _stability(original_explanation, neighborhood_explanations):
    """Inner usage, takes the explanations already calculated, instead of
    calculating it while measuring the stability. This is intended for internal
    usage, and can be used when there is a lot of repeated explanations
    being made.

    To better understand this method arguments, check how it is called by
    the public method.
    """
    return np.mean(np.power(
        _norm_p2(neighborhood_explanations - original_explanation), 2))


def stability(explainer, x, neighborhood):
    """Stability function.

    Takes as argument an explanation method, a single observation
    x of shape (n_features, ), and the neighborhood as a matrix of
    shape (n_neighbors, n_features), where each line is a sampled
    neighbor and each column is the feature value of the sample.

    Returns the mean squared p2-norm of the difference between the
    original explanation and every sampled neighbor.
    """
    
    original_explanation = explainer.explain_local(x)
    neighborhood_explanations = explainer.explain_local(neighborhood)

    # OBS: explainers should return protected explanations to avoid numeric
    # errors in the metrics. This should be implemented for every explainer.
    # the base class has a simple treatment.
    return _stability(original_explanation, neighborhood_explanations)


def _jaccard_index(As, B):
    """Method to calculate the ratio of the intersection
    over the union of two sets. This is known as Jaccard
    index and ranges from 0 to 1, measuring how simmilar
    the two sets are. A value equals to 1 means that the
    sets are identical (remembering that sets does not
    have order relations between its elements), and a
    value equals to 0 means that they are completely
    different.

    As is an array of importances (n_samples, n_features),
    and B is a 1-element importance matrix.

    Returns an array (n_obs, 1) with the index of jaccard for each observation
    """

    jaccard_indexes = np.zeros(As.shape[0])

    for i in range(As.shape[0]):
        jaccard_indexes[i] = \
            np.intersect1d(As[i, :], B).size / np.union1d(As[i, :], B).size
    
    return jaccard_indexes.reshape(-1, 1)


def _get_k_most_important(explanation, k):
    """Method that takes an array of explanation of shape
    (n_obs, n_features) and
    returns the index of the k most important (highest)
    values in the array.

    and an integer k representing the size of the subset,
    k <= len(explanations).

    Returns a python built-in set containing the indexes
    of the k highest values.
    """

    # Reversing the order so its in descending order
    order = np.argsort(explanation, axis=1)[::-1].astype(int)

    return order[:, :k]


def _jaccard_stability(original_subset, neighborhood_subset):
    """Inner usage, takes the explanations subsets already calculated,
    instead of calculating it while measuring the Jaccard stability. This is
    intended for internal usage, and can be used when there is a lot of
    repeated explanations being made.

    To better understand this method arguments, check how it is called by
    the public method.
    """
    
    return np.mean(_jaccard_index(original_subset, neighborhood_subset))


def jaccard_stability(explainer, x, neighborhood, k=1):
    """Jaccard adaptation Stability function.

    Takes as argument an explanation method, a single observation
    x of shape (n_features, ), the neighborhood as a matrix of
    shape (n_neighbors, n_features), and the size of the subset being
    considered k

    Returns the mean Jaccard Index between the original sample
    and all neighbors, considering how similar the k most important
    subset of features between the explanation of the original data
    and its neighbors.
    """

    original_subset     = _get_k_most_important(
        explainer.explain_local(x), k)
        
    neighborhood_subset = _get_k_most_important(
        explainer.explain_local(neighborhood), k)

    return _jaccard_stability(original_subset, neighborhood_subset)


def _infidelity(original_prediction, original_explanation,
    neighborhood_prediction, neighborhood_perturbation):
    """Inner usage, takes the explanations subsets already calculated,
    instead of calculating it while measuring the Jaccard stability. This is
    intended for internal usage, and can be used when there is a lot of
    repeated explanations being made.

    To better understand this method arguments, check how it is called by
    the public method.
    """

    return np.mean(np.power(
            np.dot(neighborhood_perturbation, original_explanation) - 
            (original_prediction - neighborhood_prediction), 2))


def infidelity(explainer, x, neighborhood):
    """Infidelity measure.

    Takes as argument an explanation method, a single observation
    x of shape (n_features, ), the neighborhood as a matrix of
    shape (n_neighbors, n_features), and the size of the subset being
    considered k

    Returns the infidelity measure.
    """

    original_explanation      = np.squeeze(explainer.explain_local(x))
    original_prediction       = explainer.predictor.predict(x)
    neighborhood_prediction   = explainer.predictor.predict(neighborhood)
    neighborhood_perturbation = (x - neighborhood)

    return _infidelity(original_prediction, original_explanation,
        neighborhood_prediction, neighborhood_perturbation)