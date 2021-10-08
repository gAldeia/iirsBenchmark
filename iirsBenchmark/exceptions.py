# Author:  Guilherme Aldeia
# Contact: guilherme.aldeia@ufabc.edu.br
# Version: 1.0.0
# Last modified: 08-20-2021 by Guilherme Aldeia

"""
Simple exception that is raised by explainers when they don't support local
or global explanations, or when they are not model agnostic. This should be
catched and handled in the experiments.
"""

class NotApplicableException(Exception):
    def __init__(self, message=""):
        self.message = message