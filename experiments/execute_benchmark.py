# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 10-16-2021 by Guilherme Aldeia


"""
Script to reproduce the experiments presented in the paper.

It is not advisable to abruptly interupt the experiment or copy/rename/move the
files during execution, this may lead to file corruption.
"""


import numpy  as np
import pandas as pd

import iirsBenchmark.explainers as explainers
import iirsBenchmark.regressors as regressors

from multiprocessing        import Pool, cpu_count
from itertools              import product
from iirsBenchmark.feynman  import Feynman_regressor
from _files_workers_control import (setup_environment, 
    worker_gridsearch, worker_experiment, worker_groundtruth)


# Where to find the feynman data sets and where to save the results
datasets_path = '../datasets'
results_path  = '../results'


def run_workers_in_chunks(chunk_size, configurations, worker):
    """Auxiliary method to take a chunk size, a list of configurations 
    and a worker that takes the configurations as arguments.

    This method will divide the configurations in chunks in a random order
    (this makes easier to track the results of the experiments while it is
    not finishet yet), then will perform the parallel processing in chunks.

    It seems that, for some explainers that takes long time to run, the
    python/OS/processing module ends up considering that there is an iddle 
    process in the pool and starts to execute another experiment. This is
    an alternative if you are experiencing this kind of problems.
    """
    
    # The parallel processing, chunck sizes, and many other factors make
    # the time measurements really untrustfull to make any conclusion.
    # To properly evaluate the time consumption of explainers,
    # it is better to redesign the experiments to avoid this problem
    # (or run in 1 cpu on a machine without any other processes running).

    configurations = np.array(configurations)

    # shuffle this list to avoid consecutive regressor experiments
    np.random.shuffle(configurations)

    number_of_chunks = np.ceil(
        (len(configurations)-chunk_size)/chunk_size ).astype(int)+1

    # Separating in chuncks so multiprocessing doesn't create
    # more processes than the pool size when there is a lot of blocking
    # and iddle cores.
    for i in range(number_of_chunks):
        beginIndex = i*chunk_size
        finalIndex = min(beginIndex+chunk_size, len(configurations))

        p_pool = Pool(chunk_size)

        # map is a blocking method
        p_pool.starmap(worker, configurations[beginIndex:finalIndex])
        
        p_pool.close()
        p_pool.join()


def run_workers_in_pool(cpus_to_use, configurations, worker):
    """Alternative to the previous process if you think the chunks are slower.
    """

    configurations = np.array(configurations)

    # shuffle this list to avoid consecutive regressor experiments
    np.random.shuffle(configurations)

    p_pool = Pool(cpus_to_use)
    p_pool.starmap(worker, configurations)
    
    p_pool.close()
    p_pool.join()


if __name__ == '__main__':

    # Reading the data sets names
    ds_names = pd.read_csv(
        '../datasets/FeynmanEquations.csv')['Filename'].values

    # Creating the folder structure in the results path
    setup_environment(results_path)

    # Will use half of the available cpus
    available_cpus = cpu_count()
    cpus_to_use = np.maximum(available_cpus//2, 1)
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

        # We use a list of all explainers to use every explainer to 
        # explain the fitted regressor
        groundtruth_configurations = list(product(
            ds_names,                        # List of data set names
            [Feynman_regressor],             # List with single item: feynman regressor
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
    if False: # set to False to skip, set to True to execute
        
        # gridsearch should be completely executed before the final experiments
        # to avoid simultaneous gridsearch evaluation on the same 
        # dataset-regressor. The final experiments can do the gridsearch 
        # without this step if the experiments are executed on a single process.

        print("Executing the gridsearch for all regressors")

        gridsearch_configurations = list(product(
            ds_names,                        # List of data set names
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
            ds_names,                        # List of data set names
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
            ds_names,                        # List of data set names
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
                
        run_workers_in_chunks(
            cpus_to_use, experiment_configurations, worker_experiment)        

    # finished all experiments! ------------------------------------------------
    print("Done =)")