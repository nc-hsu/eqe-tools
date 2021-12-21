"""
module containing functions for implementing the capacity spectrum method
for nonlinear static analysis

references:
Fajfar 2000
Freeman 1998
"""

import numpy as np
from pathlib import Path


def load_pushover(file_name, folder_path, header=True):
    
    """ loads a two column csv file representing the pushover curve
        of a structure. The following assumptions are made:
        (1) the first column is the base shear data
        (2) the second column is the displacement data

        Args:
            file_name (str): name of file incl. file extension
            folder_path (str/Path): path to folder where data is stored
            header (bool, optional): the file contains a header row.
                                    Defaults to True.

        Returns:
            [numpy_array]: Baseshear and Displacement vectors
    """
    
    folder_path = Path(folder_path)
    file_path = folder_path / file_name 

    with open(file_path, 'r') as file:
        if header:
            data = np.loadtxt(file, skiprows=1, delimiter=',')
        else:
            data = np.loadtxt(file, delimiter=',')

    V = data[:,0]
    D = data[:,1]
    
    return V, D


def mdof_to_sdof(V, D, m, phi):
    # TODO 
    # converts the force (V) and displacement (D) vectors for an
    # mdof system into an sdof system using the procedure described
    # in Fajfar 2000
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

    