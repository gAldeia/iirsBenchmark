import pandas as pd
import numpy  as np

from glob import glob

# https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html
from smt.sampling_methods import LHS

import os.path

# lambda functions with implementations of the equations
from iirsBenchmark.feynman._FeynmanEquations import feynmanPyData

n_samples = 1000
LHS_samples = 30


# This script expect you to have downloaded the Feynman datasets and FeynmanEquations.csv
# from https://space.mit.edu/home/tegmark/aifeynman.html

feynman_data = pd.read_csv('FeynmanEquations.csv')


if __name__ == '__main__':
    # iterating over the folder where the original data is
    for feynman_file in glob('./Feynman_with_units/*'):
        print(feynman_file)

        # Checking if this data set was already processed
        if os.path.isfile(feynman_file.replace('Feynman_with_units', 'train') +'_UNI.csv'):
            continue

        data = pd.read_csv(feynman_file, sep='\s+', index_col=None, header=None) 

        # Loading informations about the dataset
        feynman_info = feynman_data[feynman_data['Filename'] == feynman_file.replace('./Feynman_with_units/', '')]

        var_names = feynman_info.loc[:, [col for col in feynman_info.columns if '_name' in col]].dropna(axis=1).values[0]
        output_name = feynman_info['Output'].values[0]

        # Selecting fewer samples from the original data
        selected = np.random.choice(len(data), size=n_samples, replace=False)
        
        # Statistics of the selected samples
        print(data.iloc[selected].describe())

        # saving THE train data
        data.iloc[selected].to_csv(
            feynman_file.replace('Feynman_with_units', 'train') +'_UNI.csv',
            sep=',',
            header=list(var_names) + [output_name],
            index=False
        )

        # obtaining the intervals for each variable
        xlimits = data.iloc[selected].describe().loc[['min', 'max']].values.T

        # LHS sampling
        sampling = LHS(xlimits=xlimits)

        x = np.array(sampling(LHS_samples))
        
        # using the original lambda function to create the y values
        original_f = feynmanPyData[
            feynman_file.replace('./Feynman_with_units/', '')
        ]['python function']
        
        for i in range(x.shape[0]):
            x[i, -1] = original_f(x[i, :-1])
        
        # Saving the test data
        pd.DataFrame(x, columns=list(var_names) + [output_name]).to_csv(
            feynman_file.replace('Feynman_with_units', 'test') +'_LHS.csv',
            sep=',',
            header=list(var_names) + [output_name],
            index=False
        )