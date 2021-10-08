# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

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

    # Explanation quality
    'stability', 'jaccard_stability', 'infidelity',

    # Comparison between two explanations (intended to be used between ground
    # truth and a predictor with the same explainer)
    'cos_similarity'
]


def RMSE(y, yhat):

    return mean_squared_error(y, yhat, squared=False)


def R2 (y, yhat):
    
    return r2_score(y, yhat)


def cos_similarity(y, yhat):
    
    return cosine_similarity(y, yhat)


def _norm_p2(vector):
    """p2 norm of a vector.

    the vector should be an array of shape (n_obs, n_samples).
    
    vai ser retornada uma matriz (n_obs, 1) com a norma p2
    calculada para cada observação
    """

    return np.sqrt(np.sum(
        np.abs(np.power(vector, 2)), axis=1
    )).reshape(-1, 1)


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


def stability(explainer, x, neighborhood):
    """Stability function.

    Takes as argument an explanation method, a single observation
    x of shape (n_features, ), and the neighborhood as a matrix of
    shape (n_neighbors, n_features), where each line is a sampled
    neighbor and each column is the feature value of the sample.

    Returns the mean squared p2-norm of the difference between the
    original explanation and every sampled neighbor.
    """
    
    original_explanation = explainer(x)

    # OBS: explainers should return protected explanations to avoid numeric
    # errors in the metrics. This should be implemented for every explainer.
    # the base class has a simple treatment.
    return np.mean(np.power(
        _norm_p2(explainer(neighborhood) - original_explanation), 2))


def _jaccard_index(As, B):
    """Method to calculate the ratio of the intersection
    over the union of two sets. This is known as Jaccard
    index and ranges from 0 to 1, measuring how simmilar
    the two sets are. A value equals to 1 means that the
    sets are identical (remembering that sets does not
    have order relations between its elements), and a
    value equals to 0 means that they are completely
    different.

    As é uma matriz de importâncias (n_samples, n_features),
    e B é uma matriz de importância de 1 elemento.
    
    Vai ser retornada uma matriz (n_obs, 1) com o índice de 
    jaccard para cada observação
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
    order = np.argsort(explanation, axis=1)[::-1]

    return order[:, :k]


def jaccard_stability(explainer, x, neighborhood, k):
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
    original_explanation = explainer(x)
    
    original_jaccard_set = _get_k_most_important(original_explanation, k)

    return np.mean(_jaccard_index(_get_k_most_important(
        explainer(neighborhood), k), original_jaccard_set))


def infidelity(explainer, predictor, x, neighborhood):
    
    original_explanation = explainer(x)
    original_prediction  = predictor.predict(x)
    
    return np.mean(np.power(
            np.dot((x - neighborhood), np.squeeze(original_explanation)) - 
            (original_prediction - predictor.predict(neighborhood)), 2))