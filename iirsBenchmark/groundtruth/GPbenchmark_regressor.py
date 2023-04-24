# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 21-11-2021 by Guilherme Aldeia

"""
Implementation of a regressor that takes as argument the name of the 
Feynman equation label and behaves like the original
physical equation used to create the data. 

This is intended to be used as a ground truth regressor, which is ideally the
best regressor possible, since it uses the original equation (regardless of 
it having interactions or transformations).

It is also possible to use this regressor as a sanity check.
"""


from sklearn.base             import BaseEstimator
from sklearn.base             import RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

import numpy as np

from jax import grad

from iirsBenchmark.groundtruth._GPbenchmarkEquations import gpbenchmarkPyData


class GPbenchmark_regressor(BaseEstimator, RegressorMixin):
    def __init__(self, *, equation_name, **kwargs):

        self.equation_name = equation_name

        gpbenchmarkdata = gpbenchmarkPyData[equation_name]

        self.string_expression = gpbenchmarkdata['string expression']
        self.latex_expression  = gpbenchmarkdata['latex expression' ]
        self.expressible_by_IT = gpbenchmarkdata['expressible by IT']
        self.python_function   = gpbenchmarkdata['python function'  ]


    def fit(self, X, y):
        
        X, y = check_X_y(X, y)

        # Vectorizing the prediction function, and creating an attribute that
        # ends with '_' (this tells check_is_fitted that the regressor was 
        # properly fitted) before making predictions
        self.predict_f_ = lambda X: np.array(
            [self.python_function(X[i, :]) for i in range(len(X))])

        # Useful for model specific explainers that uses information about
        # selected features
        self.selected_features_ = np.array(range(X.shape[1]))

        return self
        

    def predict(self, X):
    
        check_is_fitted(self)

        X = check_array(X)

        return self.predict_f_(X)


    def gradients(self, X):
        
        gradient_f = grad(self.python_function)

        gradients = np.zeros_like( X )
        for i in range(X.shape[0]):
            gradients[i, :] = gradient_f(X[i, :])
            
        return gradients


    def to_str(self):
        
        return self.string_expression


GPbenchmark_regressor.interpretability_spectrum = 'white-box'
GPbenchmark_regressor.stochastic_executions = False
GPbenchmark_regressor.grid_params = {

}