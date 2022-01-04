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


def load_pushover(f_name: str, folder_path: str,
                  header: bool=True) -> NDArray[Float64]:
    
    """load pushover curve into two np.arrays

    Returns:
        NDArray[Float64]: pushover curve
    """
    
    folder_path = Path(folder_path)
    file_path = folder_path / f_name 

    with open(file_path, 'r') as file:
        if header:
            data = np.loadtxt(file, skiprows=1, delimiter=',')
        else:
            data = np.loadtxt(file, delimiter=',')
    
    return data


def select_folder(title: str ='') -> str:
    """opens dialog to select folder

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


def filepaths_from_folder(folder_path: Path) -> list:
    """get filepaths of all files in folder

    Args:
        folder_path (str): path of folder

    Returns:
        list: filepaths (type: Path) of files in folder
    """

    f_names = os.listdir(folder_path)
    file_paths = [folder_path / f for f in f_names]

    return file_paths


def format_spectrum(file_path: Path, Tmax: float=4.0) -> NDArray[Float64]:
    """adds period values to response spectra files containing only 
    spectral values

    Args:
        file_path (Path): path to file
        Tmax (float, optional): max period the spectra is calculate to.
                                Defaults to 4.0.

    Returns:
        NDArray[Float64]: Nx2 array with row = period/value pairs 
    """

    spectra = np.loadtxt(file_path)
    L = len(spectra)
    Ts = np.linspace(0, Tmax, L)

    return arrays_2_mat([Ts, spectra])


def arrays_2_mat(arrays: list[NDArray[Float64]],
                 cols: bool=True) -> NDArray[Float64]:
    """reshapes numpy vectors so that they can be stacked to form NDArrays

    Args:
        arrays (list[NDArray[Float64]]): vectors to be stacked
        cols (bool, optional): use vectors as columns or rows. Defaults to True.

    Returns:
        NDArray[Float64]: matrix of stacked vectors
    """

    if cols:
        arrs = tuple([np.reshape(a, (len(a), 1)) for a in arrays])
        data = np.hstack(arrs)
    else:
        arrs = tuple([np.reshape(a, (1, len(a))) for a in arrays])
        data = np.vstack(arrs)
        
    return data 


def np_2_csv(a: NDArray[Float64], f_name: str, folder: Path) -> None:
    """saves a numpy array, a, to a csv file

    Args:
        a (NDArray[Float64]): np array to be saved
        f_name (str): name of new file without extension
        folder (Path): path to folder for new file
    """
    
    np.savetxt(folder / (f_name + '.csv'), a, delimiter=',', fmt='%0.4f')
    return


def reformat_spectra() -> None:
    """reformats the spectra files in a folder to include the period
    """
    # select folder containing spectra
    folder = Path(select_folder(title=('Select folder containing spectra'
                                       ' to be formatted')))
    # get filenames of spectra
    file_paths = filepaths_from_folder(folder)
    # select save folder for spectra
    save_folder = Path(select_folder(title=('Select folder to save reformatted'
                                            ' spectra')))
    
    # reformat spectra and save
    for f in file_paths:
        f_name = f.parts[-1][:-4]
        new_spec = format_spectrum(f)
        np_2_csv(new_spec, f_name, save_folder)

    return


def format_ADRS_spectrum(sd_file: Path, sa_file: Path,
                         Tmax: float=4.0) -> NDArray[Float64]:
    
    
    sd = np.loadtxt(sd_file)
    sa = np.loadtxt(sa_file)
    L = len(sd)
    Ts = np.linspace(0, Tmax, L)

    return arrays_2_mat([sd, sa, Ts])


def format_ADRS_spectra() -> NDArray[Float64]:

    # get sd components
    sd_folder = Path(select_folder(title='Select folder containing'
                                         ' Sd spectra'))
    sd_f_paths = filepaths_from_folder(sd_folder)
    # get sa components
    sa_folder = Path(select_folder(title='Select folder containing'
                                         ' Sa spectra'))
    sa_f_paths = filepaths_from_folder(sa_folder)
    # select save folder
    save_folder = Path(select_folder(title=('Select folder to save reformatted'
                                            ' spectra')))
    
    # stack together with periods and save
    for sd, sa in zip(sd_f_paths, sa_f_paths):
        f_name = sd.parts[-1][:-4]
        new_spec = format_ADRS_spectrum(sd, sa)
        np_2_csv(new_spec, f_name, save_folder)

    return




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

    # folder = select_folder(title="dude select a folder")
    # paths = filepaths_from_folder(folder)
    # print(paths)

    # path = Path('C:/niccl/git_repos/eqe-tools/spectra/Spectrum_EQ_1RotD50.txt')
    # sp = format_spectrum(path)
    # np_2_csv(sp, 'new_spec', path.parent)
    
    format_ADRS_spectra()
    