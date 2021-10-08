# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Integrated gradients explainer.
"""

from iirsBenchmark.explainers._base_explainer import Base_explainer 
from scipy import optimize
from sklearn.utils.validation import check_array
import numpy as np


class IntegratedGradients_explainer(Base_explainer):
    def __init__(self, *, predictor, n_steps=10, **kwargs):
        super(IntegratedGradients_explainer, self).__init__(
            predictor    = predictor,
            agnostic     = True,
            local_scope  = True,
            global_scope = False
        )
        
        self.n_steps = n_steps
    
    
    def fit(self, X, y):
        self._check_fit(X, y)

        self.X_ = X
        self.y_ = y
        
        self._x_baseline = np.mean(self.X_, axis=0).reshape(1, -1)
        

    def explain_local(self, X):
        
        self._check_is_fitted()

        X = check_array(X)
            
        # Generate m_steps intervals for integral approximation below.
        alphas = np.linspace(
            start=0.0, stop=1.0, num=self.n_steps+1).reshape(-1, 1)

        integrated_gradients = np.zeros_like( X )
        for obs_idx in range(X.shape[0]):
            x = X[obs_idx, :].reshape(1, -1)
            
            delta = x - self._x_baseline

            interpolated = self._x_baseline + (delta * alphas)

            gradients = None
            if hasattr(self.predictor, "gadients"):
                
                gradients = self.predictor.gradients(interpolated)
            else:

                gradients = np.array([
                    optimize.approx_fprime(
                        x_aux,
                        lambda x: self.predictor.predict(x.reshape(1, -1)),
                        epsilon=1e-3
                    )
                    for x_aux in interpolated
                ])

            # Aproximating the integral by using trapezoidal riemann
            grads_riemann = (gradients[:-1] + gradients[1:]) / 2.0

            # Taking the mean and normalizing.
            integrated_gradients[obs_idx, :] = \
                grads_riemann.mean(axis=0) * (x - self._x_baseline)
        
        explanations = np.abs(integrated_gradients)

        return self._protect_explanation(explanations)