
# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 10-16-2021 by Guilherme Aldeia


"""
Gridsearch files
"""


import os
import time
import tempfile

import numpy  as np
import pandas as pd

from filelock import FileLock

from iirsBenchmark            import metrics
from iirsBenchmark.exceptions import NotApplicableException

from sklearn.exceptions      import NotFittedError
from sklearn.model_selection import GridSearchCV


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
        p_id, task, regressor, explainer, ds_name, rep_number))
    
    print("|{:^78s}|".format(message))
    print("+" + 78*"-" + "+")


def run_or_retrieve_gridsearch(*,
    ds_name, regressor_class, results_path, datasets_path, verbose=1):
    
    """Performs a gridsearch (if is the first time called for the given
    regressor and dataset) or retrieve the last result and returns the best
    configuration as a dictionary.
    
    The gridsearch will perform a 3-fold cv for every possible configuration
    that the regressor have in the `gridsearch_params` class attribute to the
    given data set and return the configuration that best maximizes the R2
    metric on the cross validation.
    """

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
                        regressor = regressor_class.__name__[:10],
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
                regressor = regressor_class.__name__[:10],
                explainer = '--',
                ds_name = ds_name,
                rep_number = str('--'),
                message="This gridsearch is now being executed...",
                header=True
            )
    
    # loading train data (again, pandas is not thread safe)
    with FileLock(f'{results_path}/_experiments_lock.lock'):
        train_data = pd.read_csv(
            f'{datasets_path}/train/{ds_name}_UNI.csv', sep=',', header=0,
            index_col=False).values

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    
    # Creating an instance with fixed seed for reproducibility
    reg = regressor_class(random_state=42)

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
    columns = ['dataset', 'rep', 'rmse_train', 'rmse_test',
        'r2_train', 'r2_test', 'tot_time', 'text_representation']

    regression_file = (f'{results_path}/2.regression/'
                       f'{regressor_instance.__class__.__name__}.csv')
        
    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            print_informations(
                p_id       = os.getpid(),
                task       = 'fit n eval', 
                regressor  = regressor_instance.__class__.__name__[:10],
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

        data = {
            'rmse_train' : metrics.RMSE(
                              regressor_instance.predict(X_train), y_train),
            'rmse_test' : metrics.RMSE(
                              regressor_instance.predict(X_test), y_test),
            'r2_train' : metrics.R2(
                              regressor_instance.predict(X_train), y_train),
            'r2_test' : metrics.R2(
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


def _fit_explianer(*, 
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
                regressor  = fitted_regressor_instance.__class__.__name__[:10],
                explainer  = explainer_class.__name__[:16],
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
    
    columns = ['dataset', 'rep', 'explainer', 'explanation', 'tot_time']

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
                explainer  = fitted_explainer_instance.__class__.__name__[:16],
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
                'explanation' : str(explanations).replace('\n', ' '),
                'tot_time'    : end_t - start_t
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
                regressor  = fitted_regressor_instance.__class__.__name__[:10],
                explainer  = fitted_explainer_instance.__class__.__name__[:16],
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
        
        # Explanations ---------------------------------------------------------
        explanations = None
        start_t = time.time()

        try:
            fitted_explainer_instance._check_fit(X_test, y_test)
        except NotApplicableException as e:
            explanations = ("Explainer is not agnostic and does not "
                            "support the regressor. ")
        else:
            try:
                explanations = fitted_explainer_instance.explain_local(X_test)
            except NotApplicableException as e:
                explanations = "Explainer does not support local explanations."
            except Exception as e:
                explanations = str(e)

        explanation_time = time.time() - start_t

        # Stability ------------------------------------------------------------
        stabilities = None
        start_t = time.time()
        try:
            stabilities = np.array([metrics.stability(
                fitted_explainer_instance.explain_local,
                X_test[i].reshape(1, -1),
                metrics.neighborhood(X_test[i].reshape(1, -1), X_train,
                    factor=metrics_factor, size=30),
            ) for i in range(X_test.shape[0])])

        except (NotApplicableException, NotFittedError) as e:
                stabilities = "Explainer does not support local explanations."
        except Exception as e:
            stabilities = str(e)
        
        stabilities_time = time.time() - start_t
        
        # infidelity -----------------------------------------------------------
        infidelities = None
        start_t = time.time()
        try:

            infidelities = np.array([metrics.infidelity(
                fitted_explainer_instance.explain_local,
                fitted_explainer_instance.regressor,
                X_test[i].reshape(1, -1),
                metrics.neighborhood(X_test[i].reshape(1, -1), X_train,
                    factor=metrics_factor, size=30),
            ) for i in range(X_test.shape[0])])

        except (NotApplicableException, NotFittedError) as e:
            infidelities = 'Explainer does not support local explanations.'
        except Exception as e:
            infidelities = str(e)

        infidelities_time = time.time() - start_t

        # jaccard stability ----------------------------------------------------
        jaccards = None
        start_t = time.time()
        try:

            jaccards = np.array([metrics.jaccard_stability(
                fitted_explainer_instance.explain_local,
                X_test[i].reshape(1, -1),
                metrics.neighborhood(X_test[i].reshape(1, -1), X_train,
                    factor=metrics_factor, size=30),
                k=1
            ) for i in range(X_test.shape[0])])

        except (NotApplicableException, NotFittedError) as e:
            jaccards = 'Explainer does not support local explanations.'

        except Exception as e:
            jaccards = str(e)
        
        jaccards_time = time.time() - start_t

        # Reporting all metrics
        columns = ['dataset', 'rep', 'explainer'] + \
            [f'obs_{i}' for i in range(X_test.shape[0])] + ['tot_time']

        with FileLock(f'{results_path}/_experiments_lock.lock'):

            for file, file_data, tot_time in [
                (local_file, explanations, explanation_time),
                (stability_file, stabilities, stabilities_time),
                (infidelity_file, infidelities, infidelities_time),
                (jaccard_file, jaccards, jaccards_time)
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
                    'tot_time'  : tot_time
                }
                
                for i in range(X_test.shape[0]):
                    if type(file_data) == str:
                        data[f'obs_{i}'] = file_data.replace('\n', ' ')
                    else:
                        data[f'obs_{i}'] = str(file_data[i]).replace('\n', ' ')

                local_df = local_df.append( pd.Series(data), ignore_index=True)
                local_df.to_csv(file, index=False)
 

def regressor_experiment(ds_name, regressor_class, explainer_classes,
    rep_number, results_path, datasets_path, n_local_explanations = 30,
    metrics_factor=0.001, verbose=1):
    
    """Complete experiment for a given regressor and data set.
    """

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
                f'{datasets_path}/train/{ds_name}_UNI.csv', sep=',', 
                header=0, index_col=False).values

        test_data = pd.read_csv(
                f'{datasets_path}/test/{ds_name}_LHS.csv', sep=',',
                header=0, index_col=False).values

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test = test_data[:, :-1], test_data[:, -1]

    best_configuration  = run_or_retrieve_gridsearch(
        ds_name         = ds_name,
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
            **best_configuration, random_state=rep_number),
        rep_number         = rep_number,
        results_path       = results_path
    )

    # avoid trying to explain more that exists in the train or test data sets
    n_local_explanations = \
        np.min([X_train.shape[0], X_test.shape[0], n_local_explanations])
    
    for explainer_class in explainer_classes:
        fitted_explainer = _fit_explianer(
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


def groundtruth_experiment(ds_name, feynman_regressor, explainer_classes,
    rep_number, results_path, datasets_path, n_local_explanations = 30,
    metrics_factor=0.001, verbose=1):
 
    """Complete experiment for a the original feynman equation for the data set.
    """

    if verbose:
        with FileLock(f'{tempfile.gettempdir()}/print_lock.lock'):
            message = "PROCESS {:5s} STARTED A NEW GROUNDTRUTH EXPERIMENT".\
                format(str(os.getpid()))
            
            print("+" + 78*"=" + "+")
            print("|{:^78s}|".format(message))
            print("+" + 78*"=" + "+")

    with FileLock(f'{results_path}/_experiments_lock.lock'):
        train_data = pd.read_csv(
                f'{datasets_path}/train/{ds_name}_UNI.csv', sep=',', header=0,
                index_col=False).values

        test_data = pd.read_csv(
                f'{datasets_path}/test/{ds_name}_LHS.csv', sep=',', header=0,
                index_col=False).values

        X_train, y_train = train_data[:, :-1], train_data[:, -1]
        X_test, y_test   = test_data[:, :-1], test_data[:, -1]

    # Not saving regression results. Error should always be zero.

    fitted_regressor = feynman_regressor(
        equation_name=ds_name).fit(X_train, y_train)

    n_local_explanations = \
        np.min([X_train.shape[0], X_test.shape[0], n_local_explanations])
        
    for explainer_class in explainer_classes:
        fitted_explainer = _fit_explianer(
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