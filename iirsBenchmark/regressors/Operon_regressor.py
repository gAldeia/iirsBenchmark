# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 01-22-2022 by Guilherme Aldeia

"""
Operon regressor.
"""

from operon.sklearn import SymbolicRegressor
from scipy import optimize

import numpy as np

class Operon_regressor(SymbolicRegressor):
    def __init__(self, *, 
        allowed_symbols                = 'add,sub,mul,div,constant,variable,'+
                                         'exp,log,sqrt,square,sin,cos,tanh,asin',
        crossover_probability          = 1.0,
        crossover_internal_probability = 0.9,
        mutation                       = {
            'onepoint' : 1.0, 'changevar' : 1.0, 'changefunc' : 1.0,
            'insertsubtree' : 1.0, 'replacesubtree' : 1.0,
            'removesubtree' : 1.0 },
        mutation_probability           = 0.25,
        offspring_generator            = 'basic',
        reinserter                     = 'replace-worst',
        objectives                     = ['r2'],
        max_length                     = 50,
        max_depth                      = 10,
        initialization_method          = 'btc',
        female_selector                = 'tournament',
        male_selector                  = 'tournament',
        population_size                = 1000,
        pool_size                      = None,
        generations                    = 1000,
        max_evaluations                = int(1000 * 1000),
        local_iterations               = 0,
        max_selection_pressure         = 100,
        comparison_factor              = 0,
        brood_size                     = 10,
        tournament_size                = 5,
        irregularity_bias              = 0.0,
        n_threads                      = 1,
        time_limit                     = None,
        random_state                   = None, 
        **kwargs):

        super(Operon_regressor, self).__init__(
            allowed_symbols                = allowed_symbols,
            crossover_probability          = crossover_probability,
            crossover_internal_probability = crossover_internal_probability,
            mutation                       = mutation,
            mutation_probability           = mutation_probability,
            offspring_generator            = offspring_generator,
            reinserter                     = reinserter,
            objectives                     = objectives,
            max_length                     = max_length,
            max_depth                      = max_depth,
            initialization_method          = initialization_method,
            female_selector                = female_selector,
            male_selector                  = male_selector,
            population_size                = population_size,
            pool_size                      = pool_size,
            generations                    = generations,
            max_evaluations                = max_evaluations,
            local_iterations               = local_iterations,
            max_selection_pressure         = max_selection_pressure,
            comparison_factor              = comparison_factor,
            brood_size                     = brood_size,
            tournament_size                = tournament_size,
            irregularity_bias              = irregularity_bias,
            n_threads                      = n_threads,
            time_limit                     = time_limit,
            random_state                   = random_state)
        

    def fit(self, X, y):

        super_fit = super().fit(X, y)

        self.nvars = X.shape[1]

        # Generating a string representation with notation similar to other
        # regression methods that creates a mathematical expression
        self.str_rep = self.get_model_string(precision=3)
        
        for i in range(self.nvars, 0, -1):
            self.str_rep = self.str_rep.replace(f'X{i}', f'x_{i-1}')

        # Finding which features are actually used by the model            
        self.selected_features_   = np.array(
            [i for i in range(self.nvars) if f'x_{i}' in self.str_rep])

        # Using partial effects to determine global feature importance
        self.feature_importances_ = np.mean(np.abs(
            self.gradients(X)), axis=0).reshape(1, -1)


        return super_fit
        

    def to_str(self):
        return self.str_rep


    def gradients(self, X):

        gradients = np.array([
            optimize.approx_fprime(
                x,
                lambda x: self.predict(x.reshape(1, -1)),
                epsilon=1e-3
            )
            for x in X
        ])

        return gradients


Operon_regressor.interpretability_spectrum = 'white-box'
Operon_regressor.stochastic_executions = True
Operon_regressor.grid_params = {
    "population_size" : [100, 250, 500],
    "generations"     : [100, 250, 500]
}