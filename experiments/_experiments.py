
# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.3
# Last modified: 01-01-2022 by Guilherme Aldeia


"""
Experiment files.

Every public method must take the ds_name and ds_collection as a tuple 
in the 'ds_info' argument. Private methods cares only about the ds_name.
"""


import os
import time
import tempfile

import numpy  as np
import pandas as pd

from glob import glob

from filelock import FileLock

from sklearn.exceptions      import NotFittedError
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (explained_variance_score, mean_squared_error,
                             r2_score, max_error, mean_absolute_error)
from scipy.stats import pearsonr

from iirsBenchmark            import expl_measures # explanation measures
from iirsBenchmark.exceptions import NotApplicableException

from iirsBenchmark.groundtruth  import Feynman_regressor
from iirsBenchmark.groundtruth  import GPbenchmark_regressor


def NMSE(y, yhat):
    return mean_squared_error(yhat, y) / np.var(y)


def print_informations(*, p_id, task, regressor, explainer,
    ds_name, rep_number, message, header=False):

    """Takes as argument informations about one experiment and print it
    as a box with length of 80 chars. This method is used when verbose is true
    for the experiments, and the idea is to present succint information about
    what is being executed on the terminal as stacked boxes.
    """

    if header:
        print("+" + 78*"-" + "+")    
        print("| {:^5s} | {:^10s} | {:^10s} | {:^16s} | {:^10s} | {:^10s} |".format(
            "PID", "task", "regressor", "explainer", "ds_name", "rep_number"))
        print("|" + 78*"." + "|")  

    print("| {:5d} | {:10s} | {:10s} | {:16s} | {:10s} | {:10s} |".format(
        p_id, task, regressor[:10], explainer[:16], ds_name[:10], rep_number))
    
    print("|{:^78s}|".format(message))
    print("+" + 78*"-" + "+")


def run_or_retrieve_gridsearch(*,
    ds_info, regressor_class, results_path, datasets_path, verbose=1):

    """Performs a gridsearch (if is the first time called for the given
    regressor and dataset) or retrieve the last result and returns the best
    configuration as a dictionary.
    
    The gridsearch will perform a 3-fold cv for every possible configuration
    that the regressor have in the `gridsearch_params` class attribute to the
    given data set and return the configuration that best maximizes the R2
    metric on the cross validation.
    """

    ds_name, ds_collection = ds_info

    # this method creates an auxiliary file in results_path/1.grisearch
    # to save/retrieve the best configurations. The columns are the parameters
    # that have different values to be tested.
    columns = ['dataset'] + list(regressor_class.grid_params.keys()) + ['score']

    gridsearch_file = \
        f'{results_path}/1.gridsearch/{regressor_class.__name__}.csv'

    # pandas should be manipulated inside a lock since it is not thread safe
    gridsearch_df = None
    with FileLock(f'{results_path}/_experiments_lock.lock'):

        # Try to open if it exists
        if os.path.isfile(gridsearch_file):

            gridsearch_df = pd.read_csv(gridsearch_file).copy()
        else:
            gridsearch_df = pd.DataFrame(columns=columns)
            gridsearch_df.to_csv(gridsearch_file, index=False)

        # Dataset has been already evaluated
        if len( gridsearch_df[gridsearch_df['dataset']==ds_name] )==1:
            if verbose:

                # print lock to avoid mixing prints
                with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
                    print_informations(
                        p_id = os.getpid(),
                        task = "gridsearch",
                        regressor = regressor_class.__name__,
                        explainer = '--',
                        ds_name = ds_name,
                        rep_number = str('--'),
                        message="This gridsearch was already performed and "
                                "successfully recovered",
                        header=False
                    )
        
            # Pandas uses its type sniffer when reading a data frame.
            # We'll load the data frame as objects to use python's eval()
            # function to properly retrieve the python structures of the best
            # configuration (to avoid pandas sniffer messing things)
            gridsearch_df_objects = pd.read_csv(gridsearch_file, dtype=object)

            # First we'll find the best params
            best_params_ = (
                gridsearch_df_objects[
                    gridsearch_df_objects['dataset']==ds_name].iloc[0]
            ).loc[list(regressor_class.grid_params.keys())].to_dict()

            # now we convert the entry in the data frame to a python dict
            # with the best configurations
            for k, v in best_params_.items():
                try:                
                    # We expect that the pararmeters can be interpreted as
                    # python code
                    
                    v_as_python_object = eval(v)                 
                    best_params_[k] = v_as_python_object
                except NameError as e:
                    # name error means that the parameter can be a string.
                           
                    best_params_[k] = str(v)
                except Exception as e:
                    print("An exception occured when trying to retrieve ",
                          "previous gridsearch configurations. ",
                          f"The following exception was caught {e}")
                    
                    raise
                             
            return best_params_

        # It should not have more than one best configuration for any data set
        elif len( gridsearch_df[gridsearch_df['dataset']==ds_name] )>1:

            raise Exception("Inconsistent results of gridsearch were found "
                           f"for the regressor {regressor_class.__name__} and "
                           f"dataset {ds_name}")
    
    # If the method reaches this point, then we need to perform the gridsearch
    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id = os.getpid(),
                task = "gridsearch",
                regressor = regressor_class.__name__,
                explainer = '--',
                ds_name = ds_name,
                rep_number = str('--'),
                message="This gridsearch is now being executed...",
                header=True
            )
    
    # loading train data (again, pandas is not thread safe)
    with FileLock(f'{results_path}/_experiments_lock.lock'):
        train_data = pd.read_csv(
            # Train data will be the first file that matches the name 
            # (this is because we have different sufixes in the data sets name,
            # indicating how the data was generated)
            glob(f'{datasets_path}/{ds_collection}/train/{ds_name}_*.csv')[0],
            sep=',', header=0, index_col=False
        ).values

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    
    reg = regressor_class(random_state=None)

    # gridsearch should not be parallelized to avoid nested parallelization.
    # by default gridsearch uses the regressor.score() method to find the
    # optimal configuration (default is R2 unless score is overwritten).
    grid = GridSearchCV(
        reg, regressor_class.grid_params, cv=3, verbose=0, n_jobs=1)

    grid.fit(X_train, y_train)

    best_configuration = grid.best_params_

    # Saving and returning the result    
    with FileLock(f'{results_path}/_experiments_lock.lock'):
        new_entry = pd.Series(best_configuration)
        new_entry['dataset'] = ds_name
        new_entry['score']   = grid.best_score_

        gridsearch_df = pd.read_csv(gridsearch_file).copy()

        gridsearch_df = gridsearch_df.append(new_entry, ignore_index=True)
        gridsearch_df.to_csv(gridsearch_file, index=False)

    return best_configuration


def _fit_regressor_and_save_data(*,
    ds_name, train_data, test_data, regressor_instance,
    rep_number, results_path, verbose=1):
    
    """This method performs the fit and evaluation of the regressor. The
    results are reported in the results folder inside 2.regression.
    If the regressor was already fitted and evaluated, a new instance will
    be fitted and the results will be overwritten. This happens because we 
    need the explanations to correspond to the same regressor reported, and
    if an execution is interrupted and resumed there is a chance that the
    fitted model does not correspond to the previous one. It is up to the
    experiments execution script to avoid repeated evaluations of finished
    results, as it is overwriten regardless of its completude."""

    X_train, y_train = train_data
    X_test, y_test   = test_data

    # Creating or recovering the results file
    columns = ['dataset', 'rep', 'rmse_train', 'rmse_test', 'mae_train', 'mae_test',
        'nmse_train', 'nmse_test', 'r2_train', 'r2_test',
        'max_error_train', 'max_error_test', 
        'pearson_corr_train', 'pearson_corr_test',
        'pearson_p_train', 'pearson_p_test',
        'explained_variance_score_train', 'explained_variance_score_test', 
        'tot_time', 'text_representation']

    regression_file = (f'{results_path}/2.regression/'
                       f'{regressor_instance.__class__.__name__}.csv')
        
    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id       = os.getpid(),
                task       = 'fit n eval', 
                regressor  = regressor_instance.__class__.__name__,
                explainer  = '--',
                ds_name    = ds_name,
                rep_number = str(rep_number),
                message    = "Fitting and evaluating the regressor. Old "
                             "results will be deleted",
                header=True
            )
        
    # Fitting the regressor
    start_t = time.time()
    regressor_instance.fit(X_train, y_train)
    end_t = time.time()

    # Saving the results and metrics
    regression_df   = None
    with FileLock(f'{results_path}/_experiments_lock.lock'):
        
        if os.path.isfile(regression_file):
            regression_df = pd.read_csv(regression_file).copy()
        else:
            regression_df = pd.DataFrame(columns=columns)

        if len(regression_df[(regression_df['dataset']==ds_name) &
                             (regression_df['rep']==rep_number)])>0:
        
            # Removing previous results
            regression_df = regression_df.drop(
                regression_df[
                    (regression_df['dataset']==ds_name) &
                    (regression_df['rep']==rep_number)
                ].index
            )
        pearson_train, p_train = pearsonr(
                regressor_instance.predict(X_train), y_train)

        pearson_test, p_test = pearsonr(
                regressor_instance.predict(X_test), y_test)

        data = {
            'rmse_train' : mean_squared_error(
                regressor_instance.predict(X_train), y_train, squared=False),
            'rmse_test' : mean_squared_error(
                regressor_instance.predict(X_test), y_test, squared=False),

            'mae_train' : mean_absolute_error(
                regressor_instance.predict(X_train), y_train),
            'mae_test' : mean_absolute_error(
                regressor_instance.predict(X_test), y_test),

            'nmse_train' : NMSE(
                regressor_instance.predict(X_train), y_train),
            'nmse_test' : NMSE(
                regressor_instance.predict(X_test), y_test),

            'pearson_corr_train' : pearson_train,
            'pearson_corr_test' : pearson_test,

            'pearson_p_train' : p_train,
            'pearson_p_test' : p_test,

            'max_error_train' : max_error(
                regressor_instance.predict(X_train), y_train),
            'max_error_test' : max_error(
                regressor_instance.predict(X_test), y_test),

            'explained_variance_score_train' : explained_variance_score(
                regressor_instance.predict(X_train), y_train),
            'explained_variance_score_test' : explained_variance_score(
                regressor_instance.predict(X_test), y_test),

            'r2_train' : r2_score(
                regressor_instance.predict(X_train), y_train),
            'r2_test' : r2_score(
                regressor_instance.predict(X_test), y_test),
                
            'dataset' : ds_name,
            'rep' : rep_number,
            'tot_time' : end_t - start_t,
            'text_representation' : regressor_instance.to_str()
        }

        regression_df = regression_df.append(
            pd.Series(data), ignore_index=True)

        regression_df.to_csv(regression_file, index=False)

    return regressor_instance


def _fit_explainer(*, 
    ds_name, train_data, fitted_regressor_instance, explainer_class,
    rep_number, verbose=1):

    """Fits an explainer. To do this, it is required a fitted regressor
    and the training data. The ds_name, explainer_class and rep_number
    are used only to print the informations when verbose is true
    """

    X_train, y_train = train_data

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id       = os.getpid(),
                task       = "fit explai", # only 10 characters to describe
                regressor  = fitted_regressor_instance.__class__.__name__,
                explainer  = explainer_class.__name__,
                ds_name    = ds_name,
                rep_number = str(rep_number),
                message    = "Fitting the explainer to be used to generate "
                             "local and global explanations",
                header     = True
            )

    explainer = explainer_class(predictor=fitted_regressor_instance)
    try:
        explainer.fit(X_train, y_train)
    except NotApplicableException as e:
        pass

    return explainer


def _explain_global_and_save_data(*, 
    ds_name, test_data, fitted_regressor_instance, fitted_explainer_instance,
    rep_number, results_path, verbose=1, **kwargs):

    """Takes a fitted regressor and performs global explanations to all
    applicable explainers, saving results in 3.explanation. Old results
    will be overwritten.
    
    The kwargs is used to make possible to call local and global
    explanation methods on a single loop.
    
    test_data should be a list of two tuples:
    [(X_train, y_train), (X_test, y_test)]. This way, we explain the train
    and test data sets; while 'train_data' is the data set used to train
    the regressor
    """
    
    columns = ['dataset', 'rep', 'explainer', 'explanation']

    global_folder = f'{results_path}/3.explanation/3.2.global'
    pred_name = fitted_regressor_instance.__class__.__name__

    global_file_train = f'{global_folder}/3.2.1.traindata/{pred_name}.csv'
    global_file_test  = f'{global_folder}/3.2.2.testdata/{pred_name}.csv'

    if verbose:    
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id       = os.getpid(),
                task       = "global exp",
                regressor  = pred_name[:10],
                explainer  = fitted_explainer_instance.__class__.__name__,
                ds_name    = ds_name,
                rep_number = str(rep_number),
                message    = "Explaining predictions of the regressor. Old "
                             "results will be deleted",
                header     = True
            )

    # explaining train and test data sets. Here the '_test' means it will be
    # used as test data, but we know that 'test_data' is a tuple containing
    # [(X_train, y_train), (X_test, y_test)]
    for ((X_test, y_test), global_file) in zip(
        test_data, [global_file_train, global_file_test]):

        explanations = None

        start_t = time.time()

        try:
            fitted_explainer_instance._check_fit(X_test, y_test)
        except NotApplicableException as e:
            explanations = ("Explainer is not agnostic and does not "
                            "support the regressor. ")
        else:
            try:
                explanations = fitted_explainer_instance.explain_global(X_test, y_test)[0, :]
            except NotApplicableException as e:
                explanations = "Explainer does not support global explanations."
            except Exception as e:
                explanations = str(e)
        
        end_t = time.time()
    
        # saving the results
        global_df = None 
        with FileLock(f'{results_path}/_experiments_lock.lock'):
            if os.path.isfile(global_file):
                global_df = pd.read_csv(global_file).copy()
            else:
                global_df = pd.DataFrame(columns=columns)

            # removing old results
            df_slice = global_df[
                (global_df['dataset']==ds_name) &
                (global_df['rep']==rep_number) &
                (global_df['explainer']==fitted_explainer_instance.__class__.__name__)]

            if len(df_slice)>0:
                global_df = global_df.drop(df_slice.index)

            data = {
                'dataset'     : ds_name,
                'rep'         : rep_number,
                'explainer'   : fitted_explainer_instance.__class__.__name__,
                'explanation' : str(explanations).replace('\n', ' ')
            }

            global_df = global_df.append( pd.Series(data), ignore_index=True)
            global_df.to_csv(global_file, index=False)


def _explain_local_and_save_data(*, 
    ds_name, train_data, test_data, fitted_regressor_instance, fitted_explainer_instance,
    rep_number, results_path, metrics_factor, verbose=1, **kwargs):

    """This method will perform local explanation for all observations in
    the given test_data. Old results will be overwritten.
    
    After creating all explanations, the script will also evaluate and report
    the metrics over each local explanation.
    
    Takes a fitted regressor and performs local explanations to all
    applicable explainers, saving results in 3.explanation. Old results
    will be overwritten.
    
    The kwargs is used to make possible to call local and global
    explanation methods on a single loop.
    
    test_data should be a list of two tuples:
    [(X_train, y_train), (X_test, y_test)]. This way, we explain the train
    and test data sets; while 'train_data' is the data set used to train
    the regressor
    """

    local_path_train = f'{results_path}/3.explanation/3.1.local/3.1.1.traindata/'
    local_path_test  = f'{results_path}/3.explanation/3.1.local/3.1.2.testdata/'

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id       = os.getpid(),
                task       = "local exp",
                regressor  = fitted_regressor_instance.__class__.__name__,
                explainer  = fitted_explainer_instance.__class__.__name__,
                ds_name    = ds_name,
                rep_number = str(rep_number),
                message    = "Explaining predictions of the regressor. "
                             "Old results will be deleted",
                header     = True
            )

    # Local explanations does not require the y_train
    X_train, _ = train_data

    for ((X_test, y_test), local_path) in zip(
        test_data, [local_path_train, local_path_test]):

        regressor_name = fitted_regressor_instance.__class__.__name__

        # We will do each one individually
        local_file      = f'{local_path}{regressor_name}_explanation.csv'
        stability_file  = f'{local_path}{regressor_name}_stability.csv'
        infidelity_file = f'{local_path}{regressor_name}_infidelity.csv'
        jaccard_file    = f'{local_path}{regressor_name}_jaccard.csv'
        
        # Well explain all instances locally and the neighborhood for the
        # robustness metrics 1 time only.

        explanations = None
        stabilities = None
        infidelities = None
        jaccards = None

        try:
            fitted_explainer_instance._check_fit(X_test, y_test)
        except (NotApplicableException, NotFittedError) as e:
            jaccards = stabilities = infidelities = explanations = \
                "Explainer is not agnostic and does not support the regressor."
        else:
            try:
                # Explaining the test data
                explanations = fitted_explainer_instance.explain_local(X_test)

                # For each test there will be a neighborhood to use in the 
                # measures
                nbhoods = [expl_measures.neighborhood(
                    X.reshape(1, -1), X_train, factor=metrics_factor, size=30)
                    for X in X_test]

                nbhoods_explanations = [fitted_explainer_instance.explain_local(n) 
                    for n in nbhoods]

                original_subsets = [
                    expl_measures._get_k_most_important(e.reshape(1, -1), k=1)
                    for e in explanations]

                nbhoods_subsets = [expl_measures._get_k_most_important(n, k=1) 
                    for n in nbhoods_explanations]

                original_predictions = fitted_explainer_instance.predictor.predict(X_test)

                nbhoods_predictions = [
                    fitted_explainer_instance.predictor.predict(n)
                    for n in nbhoods]

                nbhoods_perturbations = [X - nb
                    for X, nb in zip(X_test, nbhoods)]

                stabilities = np.array([expl_measures._stability(e, n)
                    for (e, n) in zip(explanations, nbhoods_explanations)
                ])

                jaccards = np.array([expl_measures._jaccard_stability(
                    o_subset, n_subset)
                    for (o_subset, n_subset) in zip(
                        original_subsets, nbhoods_subsets)
                ])

                infidelities = np.array([expl_measures._infidelity(op, oe, np, ni)
                    for (op, oe, np, ni) in zip(
                        original_predictions, explanations,
                        nbhoods_predictions, nbhoods_perturbations)
                ])

            except (NotApplicableException, NotFittedError) as e:
                jaccards = stabilities = infidelities = explanations = \
                    "Explainer does not support local explanations."

            except Exception as e:
                jaccards = stabilities = infidelities = explanations = str(e)
        
        # Reporting all metrics
        columns = ['dataset', 'rep', 'explainer'] + \
            [f'obs_{i}' for i in range(X_test.shape[0])]

        with FileLock(f'{results_path}/_experiments_lock.lock'):

            for file, file_data in [
                (local_file, explanations),
                (stability_file, stabilities),
                (infidelity_file, infidelities),
                (jaccard_file, jaccards)
            ]:

                if os.path.isfile(file):
                    local_df = pd.read_csv(file).copy()
                else:
                    local_df = pd.DataFrame(columns=columns)

                # removing old results
                df_slice = local_df[
                (local_df['dataset']==ds_name) &
                (local_df['rep']==rep_number) &
                (local_df['explainer']==fitted_explainer_instance.__class__.__name__)]

                if len(df_slice)>0:
                    local_df = local_df.drop(df_slice.index)
                    
                data = {
                    'dataset'   : ds_name,
                    'rep'       : rep_number,
                    'explainer' : fitted_explainer_instance.__class__.__name__,
                }
                
                for i in range(X_test.shape[0]):
                    if type(file_data) == str:
                        data[f'obs_{i}'] = file_data.replace('\n', ' ')
                    else:
                        data[f'obs_{i}'] = str(file_data[i]).replace('\n', ' ')

                local_df = local_df.append( pd.Series(data), ignore_index=True)
                local_df.to_csv(file, index=False)
 

def regressor_experiment(ds_info, regressor_class, explainer_classes,
    rep_number, results_path, datasets_path, n_local_explanations = 30,
    metrics_factor=0.001, verbose=1):
    
    """Complete experiment for a given regressor and data set.
    """
    ds_name, ds_collection = ds_info

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            message = "PROCESS {:5s} STARTED A NEW REGRESSOR EXPERIMENT".\
                format(str(os.getpid()))
            
            print("+" + 78*"=" + "+")
            print("|{:^78s}|".format(message))
            print("+" + 78*"=" + "+")

    # loading data
    with FileLock(f'{results_path}/_experiments_lock.lock'):
        train_data = pd.read_csv(
                glob(f'{datasets_path}/{ds_collection}/train/{ds_name}_*.csv')[0],
                sep=',', header=0, index_col=False).values

        test_data = pd.read_csv(
                glob(f'{datasets_path}/{ds_collection}/test/{ds_name}_*.csv')[0],
                sep=',', header=0, index_col=False).values

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

    best_configuration  = run_or_retrieve_gridsearch(
        ds_info         = ds_info,
        regressor_class = regressor_class, 
        results_path    = results_path,
        datasets_path   = datasets_path,
        verbose         = 1
    )

    fitted_regressor = _fit_regressor_and_save_data(
        ds_name            = ds_name, 
        train_data         = (X_train, y_train), 
        test_data          = (X_test, y_test), 
        regressor_instance = regressor_class(
            **best_configuration, random_state=None),
        rep_number         = rep_number,
        results_path       = results_path
    )

    # avoid trying to explain more that exists in the train or test data sets
    n_local_explanations = \
        np.min([X_train.shape[0], X_test.shape[0], n_local_explanations])
    
    for explainer_class in explainer_classes:
        fitted_explainer = _fit_explainer(
            ds_name                   = ds_name,
            train_data                = (X_train, y_train),
            fitted_regressor_instance = fitted_regressor,
            explainer_class           = explainer_class,
            rep_number                = rep_number
        )

        for explain_scope_funcion in \
            [_explain_global_and_save_data, _explain_local_and_save_data]:

            explain_scope_funcion(
                ds_name    = ds_name,
                train_data = (X_train, y_train),
                test_data  = [ #tuple with train and test data (in this order)
                    (X_train[:n_local_explanations, :], y_train[:n_local_explanations]),
                    (X_test[:n_local_explanations, :], y_test[:n_local_explanations])
                ],
                fitted_regressor_instance = fitted_regressor,
                fitted_explainer_instance = fitted_explainer,
                rep_number                = rep_number,
                metrics_factor            = metrics_factor, 
                results_path              = results_path
            )


def groundtruth_experiment(ds_info, groundtruth_regressor, explainer_classes,
    rep_number, results_path, datasets_path, n_local_explanations = 30,
    metrics_factor=0.001, verbose=1):
 
    """Complete experiment for a dataset with groundtruth for the data set.
    """
    ds_name, ds_collection = ds_info

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            message = "PROCESS {:5s} STARTED A NEW GROUNDTRUTH EXPERIMENT".\
                format(str(os.getpid()))
            
            print("+" + 78*"=" + "+")
            print("|{:^78s}|".format(message))
            print("+" + 78*"=" + "+")

    with FileLock(f'{results_path}/_experiments_lock.lock'):
        train_data = pd.read_csv(
            glob(f'{datasets_path}/{ds_collection}/train/{ds_name}_*.csv')[0],
            sep=',', header=0, index_col=False).values

        test_data = pd.read_csv(
            glob(f'{datasets_path}/{ds_collection}/test/{ds_name}_*.csv')[0],
            sep=',', header=0, index_col=False).values

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test   = test_data[:, :-1], test_data[:, -1]

    # Not saving regression results. Error should always be zero.

    fitted_regressor = groundtruth_regressor(
        equation_name=ds_name).fit(X_train, y_train)

    n_local_explanations = \
        np.min([X_train.shape[0], X_test.shape[0], n_local_explanations])
        
    for explainer_class in explainer_classes:
        fitted_explainer = _fit_explainer(
            ds_name                   = ds_name,
            train_data                = (X_train, y_train),
            fitted_regressor_instance = fitted_regressor,
            explainer_class           = explainer_class,
            rep_number                = rep_number
        )

        for explain_scope_funcion in \
            [_explain_global_and_save_data, _explain_local_and_save_data]:

            explain_scope_funcion(
                ds_name    = ds_name,
                train_data = (X_train, y_train),
                test_data  = [ #tuple with train and test data (in this order)
                    (X_train[:n_local_explanations, :], y_train[:n_local_explanations]),
                    (X_test[:n_local_explanations, :], y_test[:n_local_explanations])
                ],
                fitted_regressor_instance = fitted_regressor,
                fitted_explainer_instance = fitted_explainer,
                rep_number                = rep_number,
                metrics_factor            = metrics_factor, 
                results_path              = results_path
            )


def exectime_experiment(ds_info, regressor_class, explainer_classes,
    rep_number, results_path, datasets_path, n_local_explanations = 30,
    metrics_factor=0.001, verbose=1):

    """Execution time experiment. Default values are the same as the
    benchmark experiment.
    """
    ds_name, ds_collection = ds_info

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            message = " PROCESS {:5s} STARTED A NEW EXECTIME EXPERIMENT ".\
                format(str(os.getpid()))
            
            print("+" + 78*"=" + "+")
            print("|{:^78s}|".format(message))
            print("+" + 78*"=" + "+")

    # loading data
    with FileLock(f'{results_path}/_experiments_lock.lock'):
        train_data = pd.read_csv(
            glob(f'{datasets_path}/{ds_collection}/train/{ds_name}_*.csv')[0],
            sep=',', header=0, index_col=False).values

        test_data = pd.read_csv(
            glob(f'{datasets_path}/{ds_collection}/test/{ds_name}_*.csv')[0],
            sep=',', header=0, index_col=False).values

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

    best_configuration  = run_or_retrieve_gridsearch(
        ds_info         = ds_info,
        regressor_class = regressor_class, 
        results_path    = results_path,
        datasets_path   = datasets_path,
        verbose         = 1
    )

    regressor_instance = regressor_class(
        **best_configuration, random_state=None)

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id       = os.getpid(),
                task       = '    fit   ', 
                regressor  = regressor_instance.__class__.__name__,
                explainer  = '--',
                ds_name    = ds_name,
                rep_number = str(rep_number),
                message    = "Fitting and evaluating the regressor. Old "
                             "results will be deleted",
                header=True
            )
        
    # Fitting the regressor
    fitted_regressor = regressor_instance.fit(X_train, y_train)

    # avoid trying to explain more that exists in the train or test data sets
    n_local_explanations = \
        np.min([X_train.shape[0], X_test.shape[0], n_local_explanations])
    
    columns = ['dataset', 'rep'] + \
              [f'{e.__name__}_global_time' for e in explainer_classes] + \
              [f'{e.__name__}_local_time' for e in explainer_classes]

    global_times = [] # We'll fill the times inside a loop and concatenate when
    local_times  = [] # creating the final data frame.

    # Now we'll explain the fitted regressor, but reporting only the time
    # each explainer took.

    for explainer_class in explainer_classes:
        # Global explanation ---------------------------------------------------
        start_t = time.time()
        fitted_explainer = _fit_explainer(
            ds_name                   = ds_name,
            train_data                = (X_train, y_train),
            fitted_regressor_instance = fitted_regressor,
            explainer_class           = explainer_class,
            rep_number                = rep_number
        )
        try:
            fitted_explainer._check_fit(X_test, y_test)
        except (NotApplicableException, NotFittedError) as e:
            # this will indicate that the explainer is not applicable
            start_t = np.nan
                            
        else:
            try:
                # Explaining the test data
                _ = fitted_explainer.explain_local(X_test)

            except (NotApplicableException, NotFittedError) as e:
                start_t = np.nan

            except Exception as e:
                raise(e) # If this happens, the user must know

        local_times.append(time.time() - start_t)

        # Local explanation ----------------------------------------------------
        start_t = time.time()
        fitted_explainer = _fit_explainer(
            ds_name                   = ds_name,
            train_data                = (X_train, y_train),
            fitted_regressor_instance = fitted_regressor,
            explainer_class           = explainer_class,
            rep_number                = rep_number
        )
        try:
            fitted_explainer._check_fit(X_test, y_test)
        except NotApplicableException as e:
            # this will indicate that the explainer is not applicable
            start_t = np.nan

        else:
            try:
                _ = fitted_explainer.explain_global(X_train, y_train)[0, :]
            except NotApplicableException as e:
                # this will indicate that the explainer is not applicable
                start_t = np.nan

            except Exception as e:
                raise(e) # If this happens, the user must know
        
        global_times.append(time.time() - start_t)

    # saving the results

    exectime_file = f'{results_path}/4.exectime/' + \
                    f'{fitted_regressor.__class__.__name__}.csv'

    results_df = None 

    with FileLock(f'{results_path}/_experiments_lock.lock'):
        if os.path.isfile(exectime_file):
            results_df = pd.read_csv(exectime_file).copy()
        else:
            results_df = pd.DataFrame(columns=columns)

        # removing old results
        df_slice = results_df[
            (results_df['dataset']==ds_name) &
            (results_df['rep']==rep_number)]

        if len(df_slice)>0:
            results_df = results_df.drop(df_slice.index)

        data = {** {
            'dataset'   : ds_name,
            'rep'       : rep_number,
        }, **{
            f'{e.__name__}_global_time': global_times[i]
            for i, e in enumerate(explainer_classes)
        }, **{
            f'{e.__name__}_local_time': local_times[i]
            for i, e in enumerate(explainer_classes)
        }}

        results_df = results_df.append(pd.Series(data), ignore_index=True)
        results_df.to_csv(exectime_file, index=False)



def exectime_experiment_groundtruth(ds_info, groundtruth_regressor, explainer_classes,
    rep_number, results_path, datasets_path, n_local_explanations = 30,
    metrics_factor=0.001, verbose=1):

    """Execution time experiment. Default values are the same as the
    benchmark experiment.
    """

    ds_name, ds_collection = ds_info

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            message = " PROCESS {:5s} STARTED A NEW EXECTIME EXPERIMENT ".\
                format(str(os.getpid()))
            
            print("+" + 78*"=" + "+")
            print("|{:^78s}|".format(message))
            print("+" + 78*"=" + "+")

    # loading data
    with FileLock(f'{results_path}/_experiments_lock.lock'):
        train_data = pd.read_csv(
            glob(f'{datasets_path}/{ds_collection}/train/{ds_name}_*.csv')[0],
            sep=',', header=0, index_col=False).values

        test_data = pd.read_csv(
            glob(f'{datasets_path}/{ds_collection}/test/{ds_name}_*.csv')[0],
            sep=',', header=0, index_col=False).values

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

    regressor_instance = groundtruth_regressor(
        equation_name=ds_name)

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id       = os.getpid(),
                task       = '    fit   ', 
                regressor  = regressor_instance.__class__.__name__,
                explainer  = '--',
                ds_name    = ds_name,
                rep_number = str(rep_number),
                message    = "Fitting and evaluating the regressor. Old "
                             "results will be deleted",
                header=True
            )
        
    # Fitting the regressor
    fitted_regressor = regressor_instance.fit(X_train, y_train)

    # avoid trying to explain more that exists in the train or test data sets
    n_local_explanations = \
        np.min([X_train.shape[0], X_test.shape[0], n_local_explanations])
    
    columns = ['dataset', 'rep'] + \
              [f'{e.__name__}_global_time' for e in explainer_classes] + \
              [f'{e.__name__}_local_time' for e in explainer_classes]

    global_times = [] # We'll fill the times inside a loop and concatenate when
    local_times  = [] # creating the final data frame.

    # Now we'll explain the fitted regressor, but reporting only the time
    # each explainer took.

    for explainer_class in explainer_classes:
        # Global explanation ---------------------------------------------------
        start_t = time.time()
        fitted_explainer = _fit_explainer(
            ds_name                   = ds_name,
            train_data                = (X_train, y_train),
            fitted_regressor_instance = fitted_regressor,
            explainer_class           = explainer_class,
            rep_number                = rep_number
        )
        try:
            fitted_explainer._check_fit(X_test, y_test)
        except (NotApplicableException, NotFittedError) as e:
            # this will indicate that the explainer is not applicable
            start_t = np.nan
                            
        else:
            try:
                # Explaining the test data
                _ = fitted_explainer.explain_local(X_test)

            except (NotApplicableException, NotFittedError) as e:
                start_t = np.nan

            except Exception as e:
                raise(e) # If this happens, the user must know

        local_times.append(time.time() - start_t)

        # Local explanation ----------------------------------------------------
        start_t = time.time()
        fitted_explainer = _fit_explainer(
            ds_name                   = ds_name,
            train_data                = (X_train, y_train),
            fitted_regressor_instance = fitted_regressor,
            explainer_class           = explainer_class,
            rep_number                = rep_number
        )
        try:
            fitted_explainer._check_fit(X_test, y_test)
        except NotApplicableException as e:
            # this will indicate that the explainer is not applicable
            start_t = np.nan

        else:
            try:
                _ = fitted_explainer.explain_global(X_train, y_train)[0, :]
            except NotApplicableException as e:
                # this will indicate that the explainer is not applicable
                start_t = np.nan

            except Exception as e:
                raise(e) # If this happens, the user must know
        
        global_times.append(time.time() - start_t)

    # saving the results

    exectime_file = f'{results_path}/4.exectime/' + \
                    f'{fitted_regressor.__class__.__name__}.csv'

    results_df = None 

    with FileLock(f'{results_path}/_experiments_lock.lock'):
        if os.path.isfile(exectime_file):
            results_df = pd.read_csv(exectime_file).copy()
        else:
            results_df = pd.DataFrame(columns=columns)

        # removing old results
        df_slice = results_df[
            (results_df['dataset']==ds_name) &
            (results_df['rep']==rep_number)]

        if len(df_slice)>0:
            results_df = results_df.drop(df_slice.index)

        data = {** {
            'dataset'   : ds_name,
            'rep'       : rep_number,
        }, **{
            f'{e.__name__}_global_time': global_times[i]
            for i, e in enumerate(explainer_classes)
        }, **{
            f'{e.__name__}_local_time': local_times[i]
            for i, e in enumerate(explainer_classes)
        }}

        results_df = results_df.append(pd.Series(data), ignore_index=True)
        results_df.to_csv(exectime_file, index=False)