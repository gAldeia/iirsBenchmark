# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Decision tree regressor. This method is considered a white-box for most 
authors.

This regressor extends the scikit-learn
[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).

beyond what the scikit model can do, this class also implements:
* `to_str()` method, that returns a string representation of the final model;
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
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# 
class DecisionTree_regressor(DecisionTreeRegressor):

    # Nossos inits devem ser iguais (ou ter o máximo de argumentos)
    # que temos na classe derivada.

    # Mudar criterion para mudar o critério de treino
    def __init__(self, *,
        splitter='best', max_depth=None, min_samples_split=2,
        min_samples_leaf=1, min_weight_fraction_leaf=0.0,
        max_features=None, random_state=None, criterion = 'mse',
        max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, ccp_alpha=0.0, **kwargs):
        
        super(DecisionTree_regressor, self).__init__(
            criterion=criterion, splitter = splitter,
            max_depth = max_depth, min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            max_features = max_features, random_state = random_state,
            max_leaf_nodes = max_leaf_nodes,
            min_impurity_decrease = min_impurity_decrease,
            min_impurity_split = min_impurity_split,
            ccp_alpha = ccp_alpha)
        

    def to_str(self):
        
        text_rep = tree.export_text(self)
        text_rep = text_rep.replace('\n', r'<br>').replace('feature_', 'x_')

        return text_rep


# IMO, decisions trees are interpretable only if they don't get too deep.
# I will use small complexity configurations for the gridsearch.

DecisionTree_regressor.stochastic_executions = True
DecisionTree_regressor.interpretability_spectrum = 'white-box'
DecisionTree_regressor.grid_params = {
    'max_depth'      : [5, 10, 15],
    'max_leaf_nodes' : [5, 10, 15],
}
