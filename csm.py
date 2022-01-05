"""
module containing functions for implementing the capacity spectrum method
for nonlinear static analysis

references:
Fajfar 2000
Freeman 1998
"""
import os
import matplotlib
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import filedialog
from nptyping import NDArray, Float64 


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


def format_spectrum(file_path: Path, Tmax: float=4.0) -> NDArray:
    """adds period values to response spectra files containing only 
    spectral values

    Args:
        file_path (Path): path to file
        Tmax (float, optional): max period the spectra is calculate to.
                                Defaults to 4.0.

    Returns:
        NDArray: Nx2 array with row = period/value pairs 
    """

    spectra = np.loadtxt(file_path)
    L = len(spectra)
    Ts = np.linspace(0, Tmax, L)

    return arrays_2_matrix([Ts, spectra])


def arrays_2_matrix(arrays: list[NDArray],
                 cols: bool=True) -> NDArray:
    """reshapes numpy vectors so that they can be stacked to form NDArrays

    Args:
        arrays (list[NDArray]): vectors to be stacked
        cols (bool, optional): use vectors as columns or rows. Defaults to True.

    Returns:
        NDArray: matrix of stacked vectors
    """

    if cols:
        arrs = tuple([np.reshape(a, (len(a), 1)) for a in arrays])
        data = np.hstack(arrs)
    else:
        arrs = tuple([np.reshape(a, (1, len(a))) for a in arrays])
        data = np.vstack(arrs)
        
    return data 


def np_2_csv(a: NDArray, f_name: str, folder: Path) -> None:
    """saves a numpy array, a, to a csv file

    Args:
        a (NDArray): np array to be saved
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
                         Tmax: float=4.0) -> NDArray:
    """combines Sa, Sd, T values into one array

    Args:
        sd_file (Path): file containing Sd values
        sa_file (Path): file containing Sa values
        Tmax (float, optional): maximum period for spectra. Defaults to 4.0.

    Returns:
        NDArray: matrix of spectra and period values
    """
    
    sd = np.loadtxt(sd_file)
    sa = np.loadtxt(sa_file)
    L = len(sd)
    Ts = np.linspace(0, Tmax, L)

    return arrays_2_matrix([sd, sa, Ts])


def format_ADRS_spectra():
    """formats multiple spectra files in a folder to include Sa and Sd
    """

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


def plot_spectrum(spectra: NDArray, format: str='ADRS') -> None:
    """plots the response spectra in one of 3 formats. ADRS, SA or SD

    Args:
        spectra (NDArray): matrix of spectra data
        format (str, optional): plot type. One of 'ADRS', 'SA' or 'SD'.
                                Defaults to 'ADRS'.
    """
    
    # TODO - add titles and make the plot look prettier
    plt.figure()
    ax = plt.gca()
    
    if format == 'ADRS':
        ax.plot(spectra[:,0], spectra[:,1])
    elif format == 'SA':
        ax.plot(spectra[:,2], spectra[:,1])
    elif format == 'SD':
        ax.plot(spectra[:,2], spectra[:,0])

    plt.show()

    return


def load_pushover(f_name: str, folder_path: Path,
                  header: bool=True) -> NDArray:
    
    """load pushover curve into two np.arrays

    Returns:
        NDArray: pushover curve
    """
    
    file_path = folder_path / f_name 

    if header:
        data = np.loadtxt(file_path, skiprows=1, delimiter=',')
    else:
        data = np.loadtxt(file_path, delimiter=',')
    
    return data


def plot_pushover(pushover: NDArray) -> None:

    # TODO - add titles and make the plot look prettier
    plt.figure()
    ax = plt.gca()
    ax.plot(pushover[:,1], pushover[:,0])
    plt.show()

    return


def plot_capacity_spectrum(capacity_spectrum: NDArray) -> None:

    # TODO - add titles and make the plot look prettier
    plt.figure()
    ax = plt.gca()
    ax.plot(capacity_spectrum[:,0], capacity_spectrum[:,1])
    plt.show()

    return


def plot_capacity_demand_spectrum(capacity_spectrum: NDArray,
                                  demand_spectrum: NDArray) -> None:

    # TODO - add titles and make the plot look prettier
    plt.figure()
    ax = plt.gca()
    ax.plot(capacity_spectrum[:,0], capacity_spectrum[:,1])
    ax.plot(demand_spectrum[:,0], demand_spectrum[:,1])
    plt.show()
    
    return


def load_spectra():
    # TODO
    return
    
    
def pushover_2_capacity_spectrum(pushover: NDArray, w: list[float], 
                                 phi: list[float]) -> NDArray:
    # converts the force (V) and displacement (D) vectors for an
    # mdof system into an sdof system using the procedure described
    # in ATC40 (1996)
    
    w = np.array(w)
    phi = np.array(phi)
    W = sum(w)
    
    PF = sum(w * phi) / (sum(w * phi ** 2))     # participation factor
    ai = PF * sum(w * phi) / W
    
    print(PF)
    print(ai)
    
    Sa = (pushover[:,0] / W) / ai
    Sd = pushover[:,1] / (PF * phi[-1])
    
    cap_spectrum = arrays_2_matrix([Sd, Sa])
    
    return cap_spectrum





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
    
    # f_path = Path('C:/niccl/Documents/RELUIS-PROJEKT/record_selection/'
    #               'IsolaGranSasso_Data/spectra_ADRS/1463/Spectrum_EQ_1RotD50.csv')
    # spectra = np.loadtxt(f_path, delimiter=',')
    # plot_spectrum(spectra, format='SD')
    folder = Path('C:/niccl/Documents/RELUIS-PROJEKT/nls_analysis_results/'
                  'pushover/par')
    pcurve = load_pushover('A1_x_FD_par.csv', folder)
    # plot_pushover(pcurve)
    w = [859.9, 3383.9, 2238.3]
    phi = [0.333, 0.667, 1.000]
    
    cap_spec = pushover_2_capacity_spectrum(pcurve, w ,phi)
    
    spectra_folder = Path('C:/niccl/Documents/RELUIS-PROJEKT/record_selection'
                          '/IsolaGranSasso_Data/spectra_ADRS/2475')
    file = 'Spectrum_EQ_1RotD50.csv'
    dem_spec = np.loadtxt(spectra_folder / file, delimiter=',')
    plot_capacity_demand_spectrum(cap_spec, dem_spec)
    