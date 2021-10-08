# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Multi Layer Perceptron regressor. This method is considered a black-box
for most authors.

This regressor extends the scikit-learn
[MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn-neural-network-mlpregressor).

beyond what the scikit model can do, this class also implements:
* `to_str()` method, that returns a string representation of the final model;
* `stochastic_executions` attribute, indicating if the model presents
    different results between different executions if the random_state is not
    setted;
* `interpretability_spectrum` attribute, with a string representing where on
    the interpretability spectrun (white, gray or black-box) this model lays;
* `grid_params` attribute, with different possible values to be used in 
    a gridsearch optimization of the method.

Although Keras provides a more rich envirornment to work with neural networks,
the experiments will be executed on a small scale MLP. The experiments
were designed to be paralellized, which is also a downside on using keras,
because it will require to be careful as keras is not fully thread-safe.
"""

from sklearn.neural_network import MLPRegressor


class MLP_regressor(MLPRegressor):
    def __init__(self, *,
        hidden_layer_sizes=100, activation='relu', solver='adam',
        alpha=0.0001, batch_size='auto', learning_rate='constant',
        learning_rate_init=0.001, power_t=0.5, max_iter=200,
        shuffle=True, random_state=None, tol=0.0001, warm_start=False,
        momentum=0.9, nesterovs_momentum=True, early_stopping=False,
        validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
        epsilon=1e-08, n_iter_no_change=10, max_fun=15000, **kwargs):

        super(MLP_regressor, self).__init__(
            hidden_layer_sizes = hidden_layer_sizes, activation = activation,
            solver = solver, alpha = alpha, batch_size = batch_size,
            learning_rate = learning_rate,
            learning_rate_init = learning_rate_init,
            power_t = power_t, max_iter = max_iter, shuffle = shuffle,
            random_state = random_state, tol = tol,
            verbose = False, warm_start = warm_start,
            momentum = momentum, nesterovs_momentum = nesterovs_momentum,
            early_stopping = early_stopping,
            validation_fraction = validation_fraction, beta_1 = beta_1,
            beta_2 = beta_2, epsilon = epsilon,
            n_iter_no_change = n_iter_no_change, max_fun = max_fun)


    def to_str(self):
        return ("scikit MLP black-box model with params: "
               f"{str(self.get_params())}")


MLP_regressor.stochastic_executions = True
MLP_regressor.interpretability_spectrum = 'black-box'
MLP_regressor.grid_params = {
    'hidden_layer_sizes' : [(50,), (50, 100, ), (100,), (100, 100, )],
    'activation'         : ['identity', 'logistic', 'tanh', 'relu']
}