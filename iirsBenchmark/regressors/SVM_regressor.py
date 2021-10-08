
# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Support Vector Machines regressor. This boosting method is considered a
black-box for most authors, although some argue that, when using a linear
kernel, the final model can be interpreted.

This regressor extends the scikit-learn
[SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html).

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


from sklearn.svm import SVR


class SVM_regressor(SVR):
    def __init__(self, *, 
        kernel='rbf', degree=3, gamma='scale', coef0=0.0,
        tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
        max_iter=- 1, random_state=None, **kwargs):
        
        super(SVM_regressor, self).__init__(
            kernel=kernel, degree=degree, gamma=gamma, coef0=coef0,
            tol=tol, C=C, epsilon=epsilon, shrinking=shrinking,
            cache_size=cache_size, verbose=False, max_iter=max_iter)
        
    
    def to_str(self):
        return ("scikit SVM black-box model with params: "
               f"{str(self.get_params())}")


SVM_regressor.stochastic_executions = False
SVM_regressor.interpretability_spectrum = 'black-box'
SVM_regressor.grid_params = {
    'kernel' : ['linear', 'rbf', 'poly'],
    'degree' : [1, 2, 3, 4]
}
