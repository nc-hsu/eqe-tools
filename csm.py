"""
module containing functions for implementing the capacity spectrum method
for nonlinear static analysis

references:
Fajfar 2000
Freeman 1998
"""
import os
import numpy as np
import tkinter as tk
from pathlib import Path
from typing import Tuple
from tkinter import filedialog
from nptyping import NDArray, Float64 


def load_pushover(file_name: str, folder_path: str,
                  header: bool=True) -> NDArray[Float64]:
    
    """load pushover curve into two np.arrays

    Returns:
        [list]: [description]
    """
    
    folder_path = Path(folder_path)
    file_path = folder_path / file_name 

    with open(file_path, 'r') as file:
        if header:
            data = np.loadtxt(file, skiprows=1, delimiter=',')
        else:
            data = np.loadtxt(file, delimiter=',')
    
    return data


def select_folder(title: str ='') -> str:
    """opens dialog for folder selection

    Args:
        title (str, optional): title of window. Defaults to ''.

    Returns:
        str: folder path 
    """
    # opens a dialog windows explorer box to allow selection
    # of a folder

    root = tk.Tk()
    root.withdraw()

    if title != None:
        path = filedialog.askdirectory(title=title)
    else:
        path = filedialog.askdirectory()

    root.destroy()

    return path


def filepaths_from_folder(folder_path):
    # obtains the filepaths of all the files in a folder

    folder = Path(folder_path)
    file_names = os.listdir(folder)
    file_paths = [folder / f for f in file_names]

    return file_paths


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

    # folder = Path('C:/niccl/git_repos/eqe-tools')
    # file = 'test_pushover.csv'

    # V, D = load_pushover(file, folder)
    
    # for v, d in zip(V[:10], D[:10]):
    #     print(v,d)

    folder = select_folder(title="dude select a folder")
    paths = filepaths_from_folder(folder)
    print(paths)
