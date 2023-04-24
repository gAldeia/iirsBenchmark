# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.1
# Last modified: 12-27-2021 by Guilherme Aldeia


"""
"""


import numpy  as np

from multiprocessing        import Pool


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