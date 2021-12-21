"""
module containing functions for implementing the capacity spectrum method
for nonlinear static analysis
"""

import numpy as np
from pathlib import Path


def load_pushover(file_name, folder_path):
    # TODO
    folder_path = Path(folder_path)
    file_path = folder_path / file_name 

    with open(file_path, 'r') as file:
        data = np.loadtxt(file, skiprows=1, delimiter=',') # first row is header

    V = data[:,0]
    D = data[:,1]
    
    return V, D

def mdof_to_sdof():
    # TODO 
    return

def load_spectra():
    # TODO
    return


if __name__ == "__main__":
    folder = Path('C:/niccl/git_repos/eqe-tools')
    file = 'test_pushover.csv'

    V, D = load_pushover(file, folder)
    
    for v, d in zip(V[:10], D[:10]):
        print(v,d)

    