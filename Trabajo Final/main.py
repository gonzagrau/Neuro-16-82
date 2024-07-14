from neuron_models.lif import test_lif
from neuron_models.adex import test_adex
from neuron_models.hodgkin_huxley import test_hyh
from neuron_models.utils import firing_rate,fit_spikes_GA
from time import time
import  numpy as np

def main():
    # test_adex()
    # test_lif()
    # test_hyh()
    t = np.linspace(0, 100, 1000)
    spike_times = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900])
    n_per_bin = 100

    start_1 = time()
    rates = firing_rate(t, spike_times, n_per_bin)
    end_1 = time()
    print(rates)
    print(f"time elaps for non vectorized function: {end_1 - start_1}")

    start_2 = time()
    #rates = vectorized_firing_rate(t, spike_times, n_per_bin)
    end_2 = time()
    print(rates)
    print(f"time elaps for vectorized function: {end_2 - start_2}")
if __name__ == '__main__':
    main()

    