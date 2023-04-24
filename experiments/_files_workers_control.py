# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 10-16-2021 by Guilherme Aldeia


"""Auxiliary methods to deal with the results files and parallel workers.

The experiments to fit and explain regression methods with explanatory 
methods are implemented in_experiments, with methods to perform and save each
tipe of explanation. 

However, properly executing all experiments in parallel processes (without
repetition) and result files management need to be implemented as simple as
possible. 

This script implements methods for controlling the parallel processes by 
setting up all the structure needed to save the results, creating a progress
tracking file and providing easy verification of finished experiments.
Also, the experiments are wrapped in a worker function to make possible to 
run experiments in parallel.

All workers, except the gridsearch worker, expects that the worker_gridsearch
have already finished its job.

Every public method must take the ds_name and ds_collection as a tuple 
in the 'ds_info' argument. Private methods cares only about the ds_name.
"""


import os

import pandas as pd

from datetime     import datetime
from filelock     import FileLock
from _experiments import (run_or_retrieve_gridsearch, exectime_experiment,
                          exectime_experiment_groundtruth,
                          regressor_experiment, groundtruth_experiment)


def setup_environment(results_path):
    """Creates the folder structure to save the results, using as root
    the given results_path. Inside the root, the tracking file and the
    parallel write lock of the results files will be created."""
 
    print("Now creating all folders and files needed to control the "
          "execution of the experiments...")

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    subfolders = [
        '1.gridsearch',
        '2.regression',
        '3.explanation/3.1.local/3.1.1.traindata',
        '3.explanation/3.1.local/3.1.2.testdata',
        '3.explanation/3.2.global/3.2.1.traindata',
        '3.explanation/3.2.global/3.2.2.testdata',
        '4.exectime',
    ]

    for subfolder in subfolders:
        if not os.path.exists(f"{results_path}/{subfolder}"):
            os.makedirs(f"{results_path}/{subfolder}")

    # Creating lock in a path that is known to all subprocesses
    open(f'{results_path}/_experiments_lock.lock', 'w+')

    # Columns of the tracking file
    columns = ['dataset', 'regressor_name', 'rep_number', 'end_time', 'finished']

    file_benchmark = f'{results_path}/_experiments_finished_executions'
    file_exectime = f'{results_path}/_exectime_finished_executions'

    for f in [file_benchmark, file_exectime]:
        tracking_df   = pd.DataFrame(columns=columns)

        if os.path.isfile(f'{f}.csv'):
            tracking_df = pd.read_csv(f'{f}.csv')
        else:
            # creating in case it does not exists
            tracking_df.to_csv(f'{f}.csv', index=False)

        _clean_unfinished_reports(results_path, f)


def _report_started_experiment(
    ds_name, regressor_name, rep_number, results_path, control_file):
    
    """Method that takes as argument the data set name, regressor, repetition
    number and the path where the results are and updates the tracking file
    to inform that the experiment with the given configurations has started.
    """

    columns = ['dataset', 'regressor_name', 'rep_number', 'end_time', 'finished']

    tracking_file = f'{results_path}/{control_file}.csv'
    tracking_df   = pd.DataFrame(columns=columns)

    with FileLock(f'{results_path}/_experiments_lock.lock'):
        if os.path.isfile(tracking_file):
            tracking_df = pd.read_csv(tracking_file)
        else:
            tracking_df.to_csv(tracking_file, index=False)

        new_entry = pd.Series({
            'dataset': ds_name,
            'regressor_name' : regressor_name,
            'end_time' : "not finished",
            'rep_number': rep_number,
            'finished' : False
        })

        tracking_df = tracking_df.append(new_entry, ignore_index=True)
        tracking_df = tracking_df.sort_values('finished', ascending=False)
        tracking_df.to_csv(tracking_file, index=False)


def _report_finished_experiment(
    ds_name, regressor_name, rep_number, results_path, control_file):
    
    """Method that takes as argument the data set name, regressor, repetition
    number and the path where the results are and updates the tracking file
    to inform that the experiment with the given configurations is now finished.
    """

    columns = ['dataset', 'regressor_name', 'rep_number', 'end_time', 'finished']

    tracking_file = f'{results_path}/{control_file}.csv'
    tracking_df   = pd.DataFrame(columns=columns)

    with FileLock(f'{results_path}/_experiments_lock.lock'):
        if os.path.isfile(tracking_file):
            tracking_df = pd.read_csv(tracking_file)
        else:
            tracking_df.to_csv(tracking_file, index=False)

        # Dropping the previous information about the experiment
        # (this should exist, since we only report a finished experiment if
        # it has been started)
        tracking_df = tracking_df.drop(tracking_df[
                (tracking_df['dataset']==ds_name) &
                (tracking_df['regressor_name']==regressor_name) &
                (tracking_df['rep_number']==rep_number) &
                (tracking_df['finished']==False)].index)

        new_entry = pd.Series({
            'dataset': ds_name,
            'regressor_name' : regressor_name,
            'end_time' : datetime.now().strftime("%d/%m/%Y-%H:%M:%S"),
            'rep_number': rep_number,
            'finished' : True
        })

        tracking_df = tracking_df.append(new_entry, ignore_index=True)
        tracking_df = tracking_df.sort_values('finished', ascending=False)
        tracking_df.to_csv(tracking_file, index=False)


def _is_finished_experiment(
    ds_name, regressor_name, rep_number, results_path, control_file):

    """Method that takes as argument the data set name, regressor, repetition
    number and the path where the results are and checks if the experiment
    with the given configurations is already finished.
    """

    tracking_file = f'{results_path}/{control_file}.csv'
    with FileLock(f'{results_path}/_experiments_lock.lock'):

        if os.path.isfile(tracking_file):
            tracking_df = pd.read_csv(tracking_file)

            return len(tracking_df[
                (tracking_df['dataset']==ds_name) &
                (tracking_df['regressor_name']==regressor_name) &
                (tracking_df['rep_number']==rep_number) &
                (tracking_df['finished']==True)])>=1
        else:
            return False


def _clean_unfinished_reports(results_path, control_file):

    """Abrupt interruptions of the experiment script can leave unfinished
    experiments in the tracking file. This method will clean them up.
    """

    columns = ['dataset', 'regressor_name', 'rep_number', 'end_time', 'finished']

    tracking_file = f'{results_path}/{control_file}.csv'
    tracking_df   = pd.DataFrame(columns=columns)

    with FileLock(f'{results_path}/_experiments_lock.lock'):
        if os.path.isfile(tracking_file):
            tracking_df = pd.read_csv(tracking_file)
        
            tracking_df = tracking_df.drop(tracking_df[
                tracking_df['finished']==False].index)

            tracking_df.to_csv(tracking_file, index=False)
            
        else:
            tracking_df.to_csv(tracking_file, index=False)


def worker_gridsearch(ds_info, regressor_class, results_path, datasets_path):

    """Worker to perform the gridsearch in parallel processes using the
    'processing' module. This worker takes as argument the data set and
    regressor to be optimized, where the results should be saved, and
    where to find the feynman data sets.
    """

    # Gridsearch results are simple enough to be checked within the
    # gridsearch method. There is no verification here.
    # However, the 'processing' package does not support a parallel map
    # with named arguments, so this worker provides this simplification.

    run_or_retrieve_gridsearch(
        ds_info         = ds_info,
        regressor_class = regressor_class,
        results_path    = results_path,
        datasets_path   = datasets_path
    )

    return


def worker_experiment(ds_info, regressor_class, explainer_classes, rep_number,
    results_path, datasets_path, n_local_explanations, metrics_factor):

    """Worker to perform one experiment in parallel processes using the
    'processing' module. This worker takes as argument the data set, the
    regressor to be fitted, a list of explainers to be used in the 
    experiment, the number of this repetition of experiments, where the results
    should be saved, where to find the feynman data sets, the number of 
    local explanations (max=100) to perform, and the neighborhood size factor.
    """
    
    ds_name, ds_collection = ds_info

    # If already finished, skip
    if _is_finished_experiment(
        ds_name, regressor_class.__name__, rep_number,
        results_path, '_experiments_finished_executions'):
        
        return
    
    # Reporting that this experiment has started
    _report_started_experiment(
        ds_name, regressor_class.__name__, rep_number,
        results_path, '_experiments_finished_executions')

    # Performing the experiment
    regressor_experiment(
        ds_info              = ds_info,
        regressor_class      = regressor_class,
        explainer_classes    = explainer_classes,
        rep_number           = rep_number,
        n_local_explanations = n_local_explanations,
        metrics_factor       = metrics_factor,
        results_path         = results_path,
        datasets_path        = datasets_path
    )

    # Updating the status of this experiment
    _report_finished_experiment(
        ds_name, regressor_class.__name__, rep_number,
        results_path, '_experiments_finished_executions')

    return


def worker_groundtruth(ds_info, groundtruth_regressor, explainer_classes, rep_number,
    results_path, datasets_path, n_local_explanations, metrics_factor):
    
    """Worker to perform one ground-truth experiment in parallel processes
    using the 'processing' module. This worker takes as argument the data set,
    the feynman regressor class, a list of explainers to be used in the 
    experiment, the number of this repetition of experiments, where the results
    should be saved, where to find the feynman data sets, the number of 
    local explanations (max=100) to perform, and the neighborhood size factor.
    """

    ds_name, ds_collection = ds_info

    if _is_finished_experiment(
        ds_name, groundtruth_regressor.__name__, rep_number,
        results_path, '_experiments_finished_executions'):
        
        return 

    _report_started_experiment(
        ds_name, groundtruth_regressor.__name__, rep_number,
        results_path, '_experiments_finished_executions')

    groundtruth_experiment(
        ds_info               = ds_info,
        groundtruth_regressor = groundtruth_regressor,
        explainer_classes     = explainer_classes,
        rep_number            = rep_number,
        n_local_explanations  = n_local_explanations,
        metrics_factor        = metrics_factor,
        results_path          = results_path,
        datasets_path         = datasets_path,
    )
    
    _report_finished_experiment(
        ds_name, groundtruth_regressor.__name__, rep_number,
        results_path, '_experiments_finished_executions')

    return


def worker_exectime(ds_info, regressor_class, explainer_classes, rep_number,
    results_path, datasets_path, n_local_explanations, metrics_factor):

    """Worker to perform one execution time measurement in parallel processes
    using the 'processing' module. This worker takes as argument the data set,
    the regressor class, a list of explainers to be used in the 
    experiment, the number of this repetition of experiments, where the results
    should be saved, where to find the feynman data sets, the number of 
    local explanations (max=100) to perform, and the neighborhood size factor.

    It will then report only the execution time when creating the global
    explanation and the n_local_explanations for the given regressor-explainer.
    """

    ds_name, ds_collection = ds_info

    if _is_finished_experiment(
        ds_name, regressor_class.__name__, rep_number,
        results_path, '_exectime_finished_executions'):
        
        return 

    _report_started_experiment(
        ds_name, regressor_class.__name__, rep_number,
        results_path, '_exectime_finished_executions')

    exectime_experiment(
        ds_info              = ds_info,
        regressor_class      = regressor_class,
        explainer_classes    = explainer_classes,
        rep_number           = rep_number,
        n_local_explanations = n_local_explanations,
        metrics_factor       = metrics_factor,
        results_path         = results_path,
        datasets_path        = datasets_path,
    )
    
    _report_finished_experiment(
        ds_name, regressor_class.__name__, rep_number,
        results_path, '_exectime_finished_executions')

    return



def worker_exectime_groundtruth(ds_info, groundtruth_regressor, explainer_classes, rep_number,
    results_path, datasets_path, n_local_explanations, metrics_factor):
    
    """Worker to perform one ground-truth experiment in parallel processes
    using the 'processing' module. This worker takes as argument the data set,
    the feynman regressor class, a list of explainers to be used in the 
    experiment, the number of this repetition of experiments, where the results
    should be saved, where to find the feynman data sets, the number of 
    local explanations (max=100) to perform, and the neighborhood size factor.
    """

    ds_name, ds_collection = ds_info

    if _is_finished_experiment(
        ds_name, groundtruth_regressor.__name__, rep_number,
        results_path, '_exectime_finished_executions'):
        
        return 

    _report_started_experiment(
        ds_name, groundtruth_regressor.__name__, rep_number,
        results_path, '_exectime_finished_executions')

    exectime_experiment_groundtruth(
        ds_info              = ds_info,
        groundtruth_regressor    = groundtruth_regressor,
        explainer_classes    = explainer_classes,
        rep_number           = rep_number,
        n_local_explanations = n_local_explanations,
        metrics_factor       = metrics_factor,
        results_path         = results_path,
        datasets_path        = datasets_path,
    )
    
    _report_finished_experiment(
        ds_name, groundtruth_regressor.__name__, rep_number,
        results_path, '_exectime_finished_executions')

    return