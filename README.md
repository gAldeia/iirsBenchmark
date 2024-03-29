# iirsBenchmark

Benchmark proposed in the paper "Interpretability in Symbolic Regression: a benchmark of Explanatory Methods using the Feynman data set", submited to [Genetic Programming and Evolvable Machines](https://www.springer.com/journal/10710).

-----


## Paper abstract

> In some situations, the interpretability of the machine learning models play a role as important as the model accuracy. This comes from the need of trusting the prediction model, verifying some of their properties or even enforcing such properties to improve fairness. To satisfy this need, many model-agnostic explainers were proposed with the goal of working with _black-box_. Most of these works are focused on classification models, even though an adaptation to regression models is usually straightforward. Regression task can be explored with techniques considered _white-boxes_ (_e.g._, linear regression) or gray boxes (_e.g._, symbolic regression), which can deliver interpretable results. The use of explanation methods in the context of regression - and, in particular, symbolic regression - is studied in this paper, coupled with different explanation methods from the literature. Experiments were performed using 100 physics equations set together with different interpretable and non-interpretable regression methods and popular explanation methods, wrapped in a module and tested through an intensive benchmark. We adapted explanation quality metrics to inspect the performance of explainers for the regression task. The results showed that, for this specific problem domain, the Symbolic Regression models outperformed all the regression models in every quality measure. Among the tested methods, Partial Effects and SHAP presented more stable results while Integrated Gradients was unstable with tree-based models. As a byproduct of this work, we released a Python library for benchmarking explanation methods with regression models. This library will be maintened by expanding it with more explainers and regressors.


## Installation

First, you need to clone the repository.

Inside the root, execute the following commands (on linux).

```bash
# make sure you have this
sudo apt-get install build-essential

# Creating a conda environment
conda env create -f environment.yml
conda activate iirs-env

# Installing Operon first, it is the only dependence that is not on pypi

# Use gcc-9 (or later)
export CC=gcc-9
export CXX=gcc-9

# clone operon
cd iirsBenchmark/regressors
git clone https://github.com/heal-research/operon
cd operon

# run cmake with options
mkdir build; cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release  -DBUILD_PYBIND=ON -DUSE_OPENLIBM=ON -DUSE_SINGLE_PRECISION=ON -DCERES_TINY_SOLVER=ON

# build
make VERBOSE=1 -j pyoperon

# install python package
make install

# (going back to root and) Executing the local installation 
cd ../../../..
make
```


## Implemented regressors

The following regressors are available in _iirsBenchmark_:

| Regressor                       	| Class name             	| Type                	| Original implementation 	|
|---------------------------------	|------------------------	|---------------------	|-----------------	|
| XGB                             	| XGB_regressor          	| Tree boosting       	| [Scikit-learn XGB](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html)   	|
| RF                              	| RF_regressor           	| Tree bagging        	| [Scikit-learn RF](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)   	|
| MLP                             	| MLP_regressor          	| Neural network      	| [Scikit-learn MLP](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)   	|
| SVM                             	| SVM_regressor          	| Vector machine      	| [Scikit-learn SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)   	|
| k-NN                            	| KNN_regressor          	| Instance method     	| [Scikit-learn KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)   	|
| SR with coefficient optmization 	| Operon_regressor       	| Symbolic method     	| [Operon framework](https://github.com/heal-research/operon/tree/master/src)         	|
| SR with IT representation       	| ITEA_regressor         	| Symbolic method     	| [ITEA](https://github.com/gAldeia/itea-python)           	|
| Linear regression               	| Linear_regressor       	| regression analysis 	| [Scikit-learn Linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)   	|
| LASSO regression                	| Lasso_regressor        	| regression analysis 	| [Scikit-learn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)   	|
| Single decision tree            	| DecisionTree_regressor 	| Decision tree       	| [Scikit-learn Decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)   	|

The nomenclature used was ``<name of the regressor in Pascalcase>_regressor``.

All regressors implemented provides a constructor with default values for all parameters, a fit and a predict method. If you are familiar with scikit, their usage should be straight-forward.

```python
from iirsBenchmark.regressors import ITEA_regressor, Linear_regressor

from sklearn import datasets

housing_data = datasets.fetch_california_housing()
X, y = housing_data['data'], housing_data['target']

linear = Linear_regressor().fit(X, y)

# if you want to specify a parameter, it should be made by named arguments.
# there is few exceptions in iirsBenchmark where arguments are positional.
itea = ITEA_regressor(popsize=75).fit(X, y)

print(itea.stochastic_executions)   # True
print(linear.stochastic_executions) # False

print(itea.to_str())   # will print a symbolic equation
print(linear.to_str()) # will print a linear regression equation
```

The regressors are used just like any scikit-learn regressor, but our implementations extends those classes by adding a few more attributes and methods in the interpretability context:

* ``stochastic_executions``: attribute indicating if the regressor have a stochastic behavior;
* ``interpretability_spectrum``: attribute with a string indicating if the regressor is considered a _white-box_, _gray-box_ or _black-box_;
* ``grid_params``: attribute with a dictionary where each key is one parameter of the regressor and the values are lists of possible values considered in the experiments;
* ``to_str()``: method that returns a string representation of a fitted regressor (if applicable).


## Implemented explainers

Several feature attribution explanatory methods were unified in this package. The available methods are displayed below:

| Explainer                      	| Class name 	| Agnostic 	| Local 	| Global 	| Original implementation 	|
|--------------------------------	|------------	|----------	|-------	|--------	|-------------------------	|
| Random explainer           	| RandomImportance_explainer           	| Y        	| Y     	| Y      	| Our implementation     	|
| Permutation Importance         	| PermutationImportance_explainer           	| Y        	| N     	| Y      	| [scikit.inspection](https://scikit-learn.org/stable/inspection.html)       	|
| Morris Sensitivity        	| MorrisSensitivity_explainer           	| Y        	| N     	| Y      	| [interpretml](https://github.com/interpretml/interpret/)             	|
| SHapley Additive exPlanations (SHAP)                           	| SHAP_explainer           	| Y        	| Y     	| Y      	| [shap](https://github.com/slundberg/shap)                    	|
| Shapley Additive Global importancE (SAGE)                           	| SAGE_explainer            	| Y        	| N     	| Y      	| [sage](https://github.com/iancovert/sage)                    	|
| Local Interpretable Model-agnostic Explanations (LIME)                           	| LIME_explainer           	| Y        	| Y     	| N      	| [lime](https://github.com/marcotcr/lime)                    	|
| Integrated Gradients           	| IntegratedGradients_explainer           	| Y        	| Y     	| N      	| Our implementation     	|
| Partial Effects (PE)           	| PartialEffects_explainer          	| N        	| Y     	| Y      	| Our implementation     	|
| Explain by Local Approximation (ELA)	| ELA_explainer           	| Y         	| Y      	| N       	| Our implementation                        	|

The naming convention is the same as the regressors, but ``<name of the explainer in Pascalcase>_explainer``.

To explain a fitted regressor (not only the ones provided in this benchmark, but any regressor that implements a predict method), you need to instanciate the explainer, fit it to the same training data used to train the regressor, and then use the methods ``explain_local`` and ``explain_global`` to obtain feature importance explanations. If the model is not agnostic, fit will raise an exception; and if it does not support local/global explanations, it will also raise an exception when the explain functions are called.

```python
from iirsBenchmark.explainers import SHAP_explainer, PermutationImportance_explainer

# you must pass the regressor as a named argument for every explainer constructor
shap = SHAP_explainer(predictor=itea).fit(X, y)

# Local explanation takes a matrix where each line is an observation, and
# returns a matrix where each line is the feature importance for the respective input.
# Single observations should be reshaped into a 2D array with x.reshape(1, -1).
local_exps  = shap.explain_local(X[5:10, :])
local_exp   = shap.explain_local(X[3].reshape(1, -1))

# Global explanation take more than one sample (ideally the whole train/test data)
# and returns a single global feature importance for each variable.
pe = PermutationImportance_explainer(predictor=itea).fit(X, y)

global_exp = pe.explain_global(X, y)

print(global_exp)
```


## Feynman regressors

As mentioned, this benchmark uses the Feynman equations compiled and provided by [ref].

The feynman equations can be used just as any regressor in the module, but takes as a required argument the data set name which the regressor should refer. Then, the created instance can be used to predict new values, using the physics equations related to the informed data set.

A table of all equations can be found [here](https://github.com/gAldeia/iirsBenchmark/blob/7815352a92d52e27dde99ec273b0d9b3c63a47c6/datasets/FeynmanEquations.csv), where the **Filename** is the column with possible data set names argument.


## Explanation robustness measures

We strongly advise to read the Section 3.1 of our paper to fully understand how this measures work, and also check their implementation in ``iirsBenchmark.expl_measures``. Although we did not propose any of this measures, we have adaptated them when implementing in _iirsBenchmark_.

Three different explanation measures were implemented:

### Stability

The intuition is to measure the degree in which the local explanation changes for a given point compared to its neighbors.

### Infidelity

The idea of infidelity is to measure the difference between two terms:

* The dot product between a significant perturbation to a given input $X$ we are trying to explain and its explanation, and
* The output observed for the perturbed point.

### Jaccard Index

The Jaccard Index measures how similar two sets are, by calculating the ratio of its intersection size by its union size.

### Usage

To use this measures you need a fitted regressor and explainer, and them only work for local explanations:

```python
from iirsBenchmark import expl_measures

# you need to provide a neighborhood to the observation being evaluated
# with those measures

obs_to_explain = X[3].reshape(1, -1)

neighbors = expl_measures.neighborhood(
    obs_to_explain, # The observation 
    X,              # Training data to calculate the multivariate normal distribution
    factor=0.001,   # spread of the neighbors
    size=30         # number of neighbors to sample
)

expl_measures.stability(
    shap,           # the explainer we want to evaluate
    obs_to_explain, # the observation to explain
    neighbors       # sampled neighbors to evaluate the metric
)
```


## Experiments

The package implements everything we need to create experiments to evaluate interpretability quality and robustness in the regression context. 

The experiments used in the paper are in `./experiments`.


## Contributing

Feel free to contact the developers with suggestions, critics, or questions. You can always raise an issue on GitHub!


## References

This package was built upon contributions of many researchers of the XAI field, as well as the scikit-learn and Operon framework for creating and fitting a regressor.

We would like to recognize the importance of their work. To get to know each depencence better, we suggest that you read the original works mentioned below.

### Explainers

* [**SHAP**] [A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html);
* [**SAGE**] [Understanding Global Feature Contributions With Additive Importance Measures](https://arxiv.org/abs/2004.00668);
* [**LIME**] ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938);
* [**Permutation Importance**] [Random Forests](https://link.springer.com/article/10.1023%2FA%3A1010933404324);
* [**Morris Sensitivity**] [Factorial Sampling Plans for Preliminary Computational Experiments](https://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804);
* [**Integrated Gradients**] [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365);
* [**ELA**] [Explaining Symbolic Regression Predictions](https://ieeexplore.ieee.org/document/9185683);
* [**Partial Effects (for symbolic regression)**] [Measuring feature importance of symbolic regression models using partial effects](https://dl.acm.org/doi/10.1145/3449639.3459302);

### Regressors

* [**Scikit-learn module**] [Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html);
* [**ITEA**] [Interaction–Transformation Evolutionary Algorithm for Symbolic Regression](https://direct.mit.edu/evco/article/29/3/367/97354/Interaction-Transformation-Evolutionary-Algorithm);
* [**Operon**] [Operon C++: an efficient genetic programming framework for symbolic regression](https://dl.acm.org/doi/10.1145/3377929.3398099);

### Measures

* [**Stability**] [Regularizing Black-box Models for Improved Interpretability](https://papers.nips.cc/paper/2020/hash/770f8e448d07586afbf77bb59f698587-Abstract.html);
* [**Infidelity**] [On the (In)fidelity and Sensitivity of Explanations](https://proceedings.neurips.cc/paper/2019/file/a7471fdc77b3435276507cc8f2dc2569-Paper.pdf);
* [**Jaccard Index**] [S-LIME: Stabilized-LIME for Model Explanation](https://arxiv.org/abs/2106.07875).


## Comments

The development of this research is still active. We plan to extend our study by including more symbolic regression methods. As for the github, we plan to build a documentation page and provide maintence for this repository.
