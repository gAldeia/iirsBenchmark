# Auxiliary functions to create the path for the
# results files and help handling the results formats.

import matplotlib.pyplot       as plt
import seaborn                 as sns
import numpy as np

import locale

from sklearn.metrics          import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


# Standard file paths
def create_global_file_path_variables(results_path):
    global local_path_train, local_path_test, \
           global_path_train, global_path_test, regression_path, \
           feynman_path, analyzed_path

    # Local explanation files
    local_path_train = f'{results_path}/3.explanation/3.1.local/3.1.1.traindata/'
    local_path_test  = f'{results_path}/3.explanation/3.1.local/3.1.2.testdata/'

    global_path_train = f'{results_path}/3.explanation/3.2.global/3.2.1.traindata/'
    global_path_test  = f'{results_path}/3.explanation/3.2.global/3.2.2.testdata/'

    regression_path = f'{results_path}/2.regression/'


def set_mpl_sns_params(abnt=False):
    sns.set_theme(style='white')

    if abnt:
        plt.rcParams['font.family'] = ['serif']
        plt.rcParams['font.serif'] = ['Times New Roman']
        
        # comma decimal separator
        locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")

        # Tell matplotlib to use the locale we set above
        plt.rcParams['axes.formatter.use_locale'] = True

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font',   size=BIGGER_SIZE)       # controls default text sizes
    plt.rc('axes',   titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes',   labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick',  labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick',  labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.ion()


def isfloat(value):
  try:
    float(value)
    return True

  except ValueError:
    return False


def convert_to_array_of_floats(x):
    """All results are saved as floats or strings representing
    an array of floats. This function will convert the strings to
    actual arrays, and single values will also be saved as arrays
    as a convention. This should be used when loading a result csv
    file to properly calculate the metrics.
    """
    
    # Strings will also enter here, but will fail in the all() check
    if hasattr(x, '__len__'):
        if all(isfloat(xprime) for xprime in x):
            return x
    
    if isfloat(x):
        return np.array([float(x)])
        
    if isinstance(x, str):
        # if it is not an array
        if x[0] != '[' and x[-1] != ']':
            return np.nan

        return np.fromstring(x[1:-1], dtype=np.float64, sep=' ')

    return np.nan


# Some quality metrics
def Cosine_similarity(yhat, y):
    return cosine_similarity(y, yhat)[0, 0]


def RMSE(yhat, y):
    return mean_squared_error(y, yhat, squared=False)


def NMSE(yhat, y):
    return mean_squared_error(yhat, y) / np.var(y)