import numpy as np
from typing import List

## CONSTANTS ##
pV = pA = pS = ps = 1e-12
nV = nA = nS = ns = 1e-9
uV = uA = uS = us = 1e-6
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

