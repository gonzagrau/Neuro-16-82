import numpy as np
from typing import List, Callable
import matplotlib.pyplot as plt

## CONSTANTS ##
pV = pA = pS = ps = 1e-12
nV = nA = nS = ns = 1e-9
uV = uA = uS = us = uF = 1e-6
mV = mA = mS = ms = 1e-3
Mohm = 1e6

## FUNCTIONS ##

def firing_rate(t: np.linspace, spike_times: np.ndarray | List[int], n_per_bin: int=10) -> np.ndarray:
    """
    Generates a numpy array with the firing rate (in spikes per time unit) of a spike train
    :param t: time samples
    :param spike_times: indeces such that a neuron fires at t[index]
    :param n_per_bin: amount of time samples per bin, defaults to 10
    :return: an array of shape(len(t)//n_bins)
    """
    t_len = len(t)
    rates = np.zeros(t_len // n_per_bin)
    sp_count = 0
    for i in range(t_len):
        if i in spike_times:
            sp_count += 1
        bin_number,  bin_t_count = divmod(i, n_per_bin)
        # if a bin is full:
        if bin_t_count == 0:
           # skip first time sample
           if i == 0:
               continue
           rates[bin_number - 1] =  sp_count
           sp_count = 0

    return rates


def ruku4(T: np.ndarray, F: Callable, X_0: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    :param np.ndarray T: time array of len N, defined as the range a:h:b
    :param np.ndarray F: array of functions of len M
    :param np.ndarray X_0: array of initial conditions at T[0]

    Uses the Runge-Kutta 4 method to solve the following system of differential equations:
    dX/dt = F(T, X(T))
    where X and F are vectors of functions of time

    :return np.ndarray: X, an array of dimensions M x N with the values of each X[i] at T[j]
    """
    h = T[1] - T[0]
    N = len(T)
    M = len(F(0, X_0, *args, **kwargs))
    X = np.zeros((N, M))
    X[0, :] = X_0

    for j in range(N - 1):
        X_j = X[j, :]
        k1 = F(T[j], X_j, *args, **kwargs)
        k2 = F(T[j] + h / 2, X_j + (h / 2) * k1, *args, **kwargs)
        k3 = F(T[j] + h / 2, X_j + (h / 2) * k2, *args, **kwargs)
        k4 = F(T[j] + h, X_j + h * k3, *args, **kwargs)
        X[j + 1, :] = X_j + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return X


def plot_voltage(t: np.ndarray, V: np.ndarray, t_units: float, v_units: float) -> None:
    """
    Plots voltage vs. time. Params are self explainatory
    """
    fig, ax = plt.subplots()
    ax.plot(t / t_units, V / v_units, 'k-', linewidth=0.5)
    ax.set_xlabel('t [ms]')
    ax.set_ylabel('V [mV]')
    plt.show()