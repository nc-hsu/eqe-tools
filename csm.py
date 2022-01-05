"""
module containing functions for implementing the capacity spectrum method
for nonlinear static analysis

references:
Fajfar 2000
Freeman 1998
"""
import os
from tkinter.constants import N
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from pathlib import Path
from tkinter import filedialog
from nptyping import NDArray, Float64
from numpy.core.shape_base import vstack 

GRAVITY = 9.81  # m/s²


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


def find_index(arr, val, comp: str='>', n: int=1) -> int:

    if comp == '>':
        idx = np.where((arr > val) == True)[0][n-1]
    elif comp == '>=':
        idx = np.where((arr >= val) == True)[0][n-1]
    elif comp == '<':
        idx = np.where((arr < val) == True)[0][n-1]
    elif comp == '<=':
        idx = np.where((arr <= val) == True)[0][n-1]
    elif comp == '==':
        idx = np.where((arr == val) == True)[0][n-1]
    
    return idx


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


def load_po(f_name: str, folder_path: Path,
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


def plot_po(po: NDArray) -> None:

    # TODO - add titles and make the plot look prettier
    plt.figure()
    ax = plt.gca()
    ax.plot(po[:,1], po[:,0])
    plt.show()

    return


def plot_capacity_spectrum(cs: NDArray) -> None:

    # TODO - add titles and make the plot look prettier
    plt.figure()
    ax = plt.gca()
    ax.plot(cs[:,0], cs[:,1])
    plt.show()

    return


def plot_cs_ds(cs: NDArray,
                                  ds: NDArray) -> None:

    # TODO - add titles and make the plot look prettier
    plt.figure()
    ax = plt.gca()
    ax.plot(cs[:,0], cs[:,1])
    ax.plot(ds[:,0], ds[:,1])
    plt.show()
    
    return

    
def po_2_capacity_spectrum(po: NDArray, w: list[float], 
                                 phi: list[float]) -> NDArray:
    # converts the force (V) and displacement (D) vectors for an
    # mdof system into an sdof system using the procedure described
    # in ATC40 (1996)
    
    w = np.array(w)
    phi = np.array(phi)
    W = sum(w)
    
    PF = sum(w * phi) / (sum(w * phi ** 2))     # participation factor
    ai = PF * sum(w * phi) / W
    
    Sa = (po[:,0] / W) / ai
    Sd = po[:,1] / (PF * phi[-1])
    
    cs = arrays_2_matrix([Sd, Sa])
    
    return cs


def simplify_cs_backbone(cs: NDArray ,type_: str='bi') -> NDArray:
    # produces a simplified backbone curve using:
    #        (1) bilinear approximation - 'bi'
    # TODO   (2) trilinear approximation - 'tri'
    # TODO   (3) quadrilinear approimation - 'quad'
    
    if type_ == 'bi':
        simple_cs = cs_bilinear(cs)
    
    return simple_cs


def cs_bilinear(cs: NDArray, mu: float=6, tol: float=0.0001) -> NDArray:
    """ produces bilinear approximation of the provided capacity spectrum
        Assumes the following:
           (1) elastic-perfectly plastic
           (2) areas under the curve match within tolerance
           (3) fitted intial stiffness crosses at 0.6*Fy

    Args:
        cs (NDArray): capacity spectrum
        mu (float, optional): maximum ductility to compute fit. Defaults to 6.
        tol (float, optional): tolerance of difference of areas. 
                               Defaults to 0.0001.

    Returns:
        NDArray: the bilinearised capacity spectrum
    """
    Fyi = [np.max(cs[:,1])] # fist guess of yield point - assume max
    Fy_idx = np.argmax(cs[:,1])
    
    diff = 1    # diff normalised by proportion of area under bilin. curve
    count = 0
    
    while diff > tol and count < 1000:
        # determine displacement of delta_y intersection
        F_cross = 0.6 * Fyi[-1]
        D_cross = np.interp(F_cross, cs[:,1][0:Fy_idx], cs[:,0][0:Fy_idx])
        Dyi = D_cross / 0.6
        
        # maximum displacement at which to calculate the area
        Dmax = min(mu * Dyi, cs[:,0][-1])

        # determine area under bilinear curve
        A_bi = (Dyi / 2 + (Dmax - Dyi)) * Fyi[-1]
         
        # determine the area under the pushover curve
        d_idx = np.where((cs[:,0] >= Dmax) == True)[0][0]
        cs_trunc = cs[0:d_idx+1,:].copy()
        cs_trunc[-1,0] = np.interp(Dmax, cs_trunc[:,0], cs_trunc[:, 1])
        cs_trunc[-1,1] = Dmax
        A_cs = np.trapz(cs_trunc[:, 1], cs_trunc[:,0])  # Area 
        
        # check whether not the areas are within the tolerance
        diff = (A_bi - A_cs)
        if abs(diff) > tol:
            # update the guess of Fy
            if diff > 0:
                Fyi.append(Fyi[-1]-0.001*Fyi[0])
            elif diff < 0:
                Fyi.append(Fyi[-1]+0.0005*Fyi[0])

        count += 1   
          
    # final bilinear cs curve
    cs_bi = np.transpose(np.array([[0, Dyi, Dmax],
                                   [0, Fyi[-1], Fyi[-1]]]))

    return cs_bi

   
def cs_resample(cs: NDArray, delta: float=0.001, type_: str='bi') -> NDArray:
    # resamples the post yield simplified cs with points spaced by delta.
    # TODO trilinear type
    # TODO bilinear type
    
    delta_p = cs[2,0] - cs[1,0]
    n = round(delta_p / delta)
    d_max = cs[1,0] + (n * delta)
    
    if type_ == 'bi':
        ds = np.arange(cs[1,0], d_max, step=delta)
        fs = np.interp(ds, cs[:,0], cs[:,1])
        new_vals = arrays_2_matrix([ds, fs], cols=True)
        cs_re = np.vstack((cs[0,:], new_vals))
        
    return cs_re
    
 
def cs_mus(cs: NDArray) -> NDArray: 
    # yield point is always row index one
    return cs[1:,0] / cs[1,0]
    
            
def cs_Ti(cs: NDArray) -> float:
    # yield point is always row index one
    return 4 * np.pi ** 2 * cs[1,0] / cs[1,1] / GRAVITY


def cs_Teffs(Ti: float, mu: NDArray, r: float=0.0) -> NDArray:
    # TODO modify for tri and quadlinear
    return np.sqrt(mu / (1 + r * (mu - 1))) * Ti


def cs_ksis(mu: NDArray, C: float) -> NDArray:

    return 0.05 + C * (mu - 1) / (np.pi * mu)
    
    
def cs_etas(ksi: NDArray) -> NDArray:

    return np.sqrt(0.07 / (0.02 + ksi))
    

def ds_resample(ds: NDArray, Teffs: NDArray) -> NDArray:

    # find first fist row index where Teff > T
    idx = find_index(ds[:,2], Teffs[0], '>')
    # split matrix
    ds1 = ds[0:idx, :].copy()
    ds2 = ds[idx:, :].copy()
    # resample sd and sa for Teffs
    new_sd = np.interp(Teffs, ds2[:,2], ds2[:,0])
    new_sa = np.interp(Teffs, ds2[:,2], ds2[:,1])
    new_ds2 = arrays_2_matrix([new_sd, new_sa, Teffs])
    # recombine
    ds_re = vstack((ds1, new_ds2))
    
    return ds_re
    
    
   
   
def variable_damping_spectra():
    return


if __name__ == "__main__":

    # folder = Path('C:/niccl/git_repos/eqe-tools')
    # file = 'test_pushover.csv'

    # V, D = load_po(file, folder)
    
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
    po = load_po('A1_x_FD_par.csv', folder)
    # plot_pushover(pcurve)
    w = [859.9, 3383.9, 2238.3]
    phi = [0.333, 0.667, 1.000]
    
    cs = po_2_capacity_spectrum(po, w ,phi)
    
    ds_folder = Path('C:/niccl/Documents/RELUIS-PROJEKT/record_selection'
                          '/IsolaGranSasso_Data/spectra_ADRS/2475')
    ds_file = 'Spectrum_EQ_5RotD50.csv'
    ds = np.loadtxt(ds_folder / ds_file, delimiter=',')
    
    cs_bi = cs_bilinear(cs)
    cs_bi = cs_resample(cs_bi)
    
    mus = cs_mus(cs_bi)
    Ti = cs_Ti(cs_bi)
    Teffs = cs_Teffs(Ti, mus)
    
    C = 0.565
    
    ksis = cs_ksis(mus, C)
    etas = cs_etas(ksis)
    
    ds_re = ds_resample(ds, Teffs)
    print('ds shape: ', np.shape(ds))
    print('ds_re shape: ', np.shape(ds_re))
    print(len(Teffs))
    print(cs_bi)
    
    T1 = np.array([[0,0],[0.06507, 1.1655]])
    T = np.array([[0,0],[0.32007, 1.1655]])
    # plot_cs_ds(cs_bi, ds_re)
    
    # print(cs_bi)
    # print()
    # print(cs)
    # # plot_cs_ds(cs_bi, ds)
    # # plot_cs_ds(cs, ds)
    plt.figure()
    ax = plt.gca()
    ax.plot(T[:,0], T[:,1], cs_bi[:,0], cs_bi[:,1], ds_re[:,0], ds_re[:,1], T1[:,0], T1[:,1])
    ax.plot(0.0592,1.0781, marker='o', mfc='red', mec='k')
    ax.plot(cs[:,0], cs[:,1],)
    plt.show()