# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 12-27-2021 by Guilherme Aldeia


"""
Script to reproduce the experiments presented in the paper.

It is not advisable to abruptly interupt the experiment or copy/rename/move the
files during execution, this may lead to file corruption.
"""


import numpy  as np
import pandas as pd

import iirsBenchmark.explainers as explainers
import iirsBenchmark.regressors as regressors

from multiprocessing        import cpu_count
from itertools              import product
from iirsBenchmark.groundtruth  import Feynman_regressor
from iirsBenchmark.groundtruth  import GPbenchmark_regressor
from _files_workers_control import (setup_environment, 
    worker_gridsearch, worker_experiment, worker_groundtruth)
from _pool_managers         import (run_workers_in_chunks, run_workers_in_pool)


# Where to find the feynman data sets and where to save the results
datasets_path = '../datasets'
datasets_collections = ['GPbenchmark'] #['Feynman', 'GPbenchmark']
results_path  = '../results'


if __name__ == '__main__':

    # Reading the data sets names
    ds_infos = []

    for dss in datasets_collections:
        filenames = pd.read_csv(
            f'{datasets_path}/{dss}/{dss}Equations.csv')['Filename'].to_list()

        # Appending the dataset names of the collection
        ds_infos = ds_infos + [(f, dss) for f in filenames]

    # Creating the folder structure in the results path.
    setup_environment(results_path)

    # Will use half of the available cpus
    available_cpus = cpu_count()
    cpus_to_use = np.maximum(10, 1)
    print(f"Available cpus: {available_cpus}. Using {cpus_to_use}.")

    print("Experiments are about to be executed. The informations will be "
          "printed as a table. Below you will see the table header.\n"
          "PRESS ENTER TO START.")

    # press enter to start (no wonder why I decided to do that). Don't forget
    # to actually start the experiments when calling the script =]
    input()

    # Groundtruth --------------------------------------------------------------
    if False: # set to False to skip, set to True to execute

        print("Executing the explainers for the groundtruth expression")

        # Everything must be a list to work with itertools' product method.
        for dss in datasets_collections:
            filenames = pd.read_csv(
            f'{datasets_path}/{dss}/{dss}Equations.csv')['Filename'].to_list()

            # Appending the dataset names of the collection
            ds_infos = [(f, dss) for f in filenames]

            groundtruth_regressor = Feynman_regressor if dss=="Feynman" else GPbenchmark_regressor
            
            # We use a list of all explainers to use every explainer to 
            # explain the fitted regressor
            groundtruth_configurations = list(product(
                ds_infos,                        # List of data set names
                [groundtruth_regressor],             # List with single item: feynman regressor
                [[getattr(explainers, explainer) # List with single item: list of all explainer classes
                    for explainer in explainers.__all__]],
                [0],                             # List with single item: index of repetitions
                [results_path],                  # List with single item: the result path
                [datasets_path],                 # List with single item: data sets path
                [30],                            # List with single item: Number of samples to locally explain
                [0.001],                         # List with single item: Neighborhood lambda parameter
                repeat=1                         # attribute of product method
            ))

            run_workers_in_chunks(
                cpus_to_use, groundtruth_configurations, worker_groundtruth)
        
    # gridsearch ---------------------------------------------------------------
    if True: # set to False to skip, set to True to execute
        
        # gridsearch should be completely executed before the final experiments
        # to avoid simultaneous gridsearch evaluation on the same 
        # dataset-regressor. The final experiments can do the gridsearch 
        # without this step if the experiments are executed on a single process.

        print("Executing the gridsearch for all regressors")

        gridsearch_configurations = list(product(
            ds_infos,                        # List of data set names
            [getattr(regressors, regressor)  # List with all regressors classes
                for regressor in regressors.__all__],
            [results_path],                  # List with single item: the result path
            [datasets_path],                 # List with single item: data sets path
            repeat=1                         # attribute of product method
        ))

        run_workers_in_chunks(
            cpus_to_use, gridsearch_configurations, worker_gridsearch)        

    # final experiments --------------------------------------------------------
    if True: # set to False to skip, set to True to execute

        # combining stochastic and deterministic  with different 
        # number of repetitions
        
        print("Executing the final experiment for all regressors")

        experiment_configurations = list(product(
            ds_infos,                        # List of data set names
            [getattr(regressors, regressor)  # List with all stochastic regressors classes
                for regressor in regressors.__all__
                if getattr(regressors, regressor).stochastic_executions], 
            [[getattr(explainers, explainer) # List with single item: list of all explainer classes
                for explainer in explainers.__all__]],
            list(range(30)),                 # List with the index of repetitions
            [results_path],                  # List with single item: the result path
            [datasets_path],                 # List with single item: data sets path
            [30],                            # List with single item: Number of samples to locally explain
            [0.001],                         # List with single item: Neighborhood lambda parameter
            repeat=1                         # attribute of product method
        )) + list(product(
            ds_infos,                        # List of data set names
            [getattr(regressors, regressor)  # List with all non-stochastic regressors classes
                for regressor in regressors.__all__
                if not getattr(regressors, regressor).stochastic_executions], 
            [[getattr(explainers, explainer) # List with single item: list of all explainer classes
                for explainer in explainers.__all__]],
            [0],                             # List with single item: index of repetitions
            [results_path],                  # List with single item: the result path 
            [datasets_path],                 # List with single item: data sets path 
            [30],                            # List with single item: Number of samples to locally explain 
            [0.001],                         # List with single item: Neighborhood lambda parameter 
            repeat=1                         # attribute of product method
        ))

        experiment_configurations = list(product(
            ds_infos,                        # List of data set names
            [getattr(regressors, 'ITEA_regressor')], 
            [[getattr(explainers, explainer) # List with single item: list of all explainer classes
                for explainer in explainers.__all__]],
            list(range(30)),                 # List with the index of repetitions
            [results_path],                  # List with single item: the result path
            [datasets_path],                 # List with single item: data sets path
            [30],                            # List with single item: Number of samples to locally explain
            [0.001],                         # List with single item: Neighborhood lambda parameter
            repeat=1                         # attribute of product method
        ))

        run_workers_in_chunks(
            cpus_to_use, experiment_configurations, worker_experiment)        

    # finished all experiments! ------------------------------------------------
    print("Done =)")