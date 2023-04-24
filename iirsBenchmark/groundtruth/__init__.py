
# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Implementation of a regressor that takes as argument the name of the 
Feynman equation label and behaves like the original
physical equation used to create the data. 

This is intended to be used as a ground truth regressor, which is ideally the
best regressor possible, since it uses the original equation (regardless of 
it having interactions or transformations).
"""

from iirsBenchmark.groundtruth.Feynman_regressor     import Feynman_regressor
from iirsBenchmark.groundtruth.GPbenchmark_regressor import GPbenchmark_regressor

__all__ = ['Feynman_regressor', 'GPbenchmark_regressor']