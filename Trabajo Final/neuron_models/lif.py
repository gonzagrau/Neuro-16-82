import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Callable, Tuple, List, Dict # para hacer type hinting
from scipy.optimize import least_squares, differential_evolution
from .utils import firing_rate, plot_voltage
from .genetic_algo import GeneticAlgorithm
# Importamos las constantes de unidades
from .utils import pV, pA, pS, Mohm
from .utils import nV, nA, nS, ns
from .utils import uV, uA, uS, us
from .utils import mV, mA, mS, ms

class LIF_model(object):
    DEFAULT_PARS = {
        'V_th' : -55*mV,  # spike threshold [mV]
        'V_reset' : -75*mV,  # reset potential [mV]
        'tau_m' : 10*ms,  # membrane time constant [ms]
        'g_L' : 10*nS,  # leak conductance [nS]
        'V_init' : -65*mV,  # initial potential [mV]
        'E_L' : -65*mV,  # leak reversal potential [mV]
        'tref' : 2*ms,  # refractory time (ms)
        'V_fire' : 40*mV,  # fire potential
    }
    VALID_KEYS = DEFAULT_PARS.keys()

    def __init__(self, **kwargs):
        self.V_th = None
        self.V_reset = None
        self.tau_m = None
        self.g_L = None
        self.V_init = None
        self.E_L = None
        self.tref = None
        self.V_fire = None
        # Parameters can be set either by passing them as kwargs...
        for name, value in kwargs.items():
            if name not in LIF_model.VALID_KEYS:
                raise ValueError(f"{name} is not a valid attribute for this class")
            setattr(self, name, value)
        # ... or set by default
        for def_name, def_val in LIF_model.DEFAULT_PARS.items():
            if getattr(self, def_name) is None:
                setattr(self, def_name, def_val)


    def derivative(self, I_val: float, u: float) -> np.ndarray:
        """
        Computes the instantaneous derivative
        :param I_val: current current (hehe)
        :param u: current voltage
        :return: du/dt
        """
        s = self # alias for neat code
        du = -(u - s.E_L) + I_val / s.g_L
        return du/s.tau_m

    def simulate_trajectory(self, t: np.ndarray,
                            I_input: np.ndarray,
                            plot: bool = False,
                            t_units: float = ms,
                            v_units: float = mV) -> Tuple[np.ndarray, List[int]]:
        """
        Solves IVP to find the trajectory v(t)
        :param t: time samples
        :param I_input: input current, same shape as t
        :param plot: indicates whether to plot
        :param I_units: for plotting purposes
        :param t_units: for plotting purposes
        :param v_units: for plotting purposes
        :return: V, same shape as t, and a list of spike times
        """
        s = self
        V = np.zeros_like(t)
        dt = t[1] - t[0]
        V[0] = self.V_init
        ref_counter = 0
        spike_times = []
        # Simulamos
        for i in range(1, len(t)):
            V_next = V[i - 1] + s.derivative(I_input[i - 1], V[i - 1]) * dt

            # Caso 1: estamos en periodo refractario
            if ref_counter > 0 or V[i - 1] == s.V_fire:
                V[i] = s.V_reset
                ref_counter -= dt

            # Caso 2: disparo
            elif V_next >= s.V_th:
                V[i] = s.V_fire
                ref_counter = s.tref
                spike_times.append(i)

            # Caso 3: nada en particular, integramos
            else:
                V[i] = V_next

        if plot:
            plot_voltage(t, V, t_units, v_units)

        return V, spike_times


    def get_init_pars_2_fit(self, keys: List[str], units: List[int]) -> np.ndarray:
        """
        Initializes a parameter array to use and tweak in fitting functions
        :param keys: lists the names of parameters to tweak, e.g. 'V_ref'
        :param units: units for each parameter
        :return: initial parameters and units
        """
        init_pars = []
        for key, unit in zip(keys, units):
            init_pars.append(self.__getattribute__(key) / unit)
        init_pars = np.array(init_pars)

        return init_pars


    def update_params(self, keys: np.ndarray | List[str],
                      pars: np.ndarray,
                      units: np.ndarray | List[int]):
        """
        Updates parameters in model object
        :param keys: keys (i.e. param name) for each value
        :param pars: new values
        :param units: units for each value
        :return: None
        """
        for key, par, unit in zip(keys, pars, units):
            self.__setattr__(key, par * unit)


    def fit_spikes_GA(self, t: np.ndarray,
                      obj_spikes: np.ndarray,
                      I_input: np.ndarray,
                      n_per_bin: int=10,
                      tweak_keys: List['str'] | None=None,
                      tweak_units: List[int | float] | None=None,
                      N_iter: int=1000,
                      pop_size: int=100,
                      mut_rate: float=0.01) -> None:
        """
        Tweaks the object's parameters to fit a voltage curve using genetic algorithms
        :param t: time array
        :param obj_spikes: neuron spike readings
        :param I_input: input current
        :param n_per_bin: bins for the firing rate calculation
        :param tweak_keys: parameters to tweak
        :param tweak_units: units for said parameters
        :param pop_size: population size for GA
        :param N_iter: maximum number of algorithm iterations
        :param mut_rate: float in [0, 1) for mutation rate in GA
        :return: None, but the internal parameters are tweaked to the best fitting
        """
        if tweak_units is None:
            if tweak_keys is None:
                tweak_units = [ms, nS]
                tweak_keys = ['tau_m', 'g_L']
            else:
                tweak_units = [1 for _ in range(len(tweak_keys))]

        obj_rates = firing_rate(t, obj_spikes, n_per_bin)
        init_pars = self.get_init_pars_2_fit(tweak_keys, tweak_units)

        def fitness_function(pars):
            self.update_params(tweak_keys, pars, tweak_units)
            _, sim_spikes = self.simulate_trajectory(t, I_input)
            sim_rates = firing_rate(t, sim_spikes, n_per_bin)
            rate_error = np.mean((obj_rates - sim_rates)**2)
            # timing_error = sum([abs(t1 - t2) for t1, t2 in zip(sim_spikes, obj_spikes)])
            return 1 / (1 + rate_error)

        algo_obj = GeneticAlgorithm(pop_size, N_iter, mut_rate, fitness_function, init_pars)
        best_solution = algo_obj.genetic_algorithm()

        self.update_params(tweak_keys, best_solution, tweak_units)

    def fit_spikes(self, t: np.ndarray,
                   obj_spikes: List[int] | np.ndarray,
                   I_input: np.ndarray,
                   n_per_bin: int=10,
                   max_iter: int=1000) -> None:
        """
        Tweaks some model parameters to fit some objective spike times
        Args:
            t (np.ndarray): time array
            obj_spikes (List[int] | np.ndarray): list of indeces where spikes happen
            I_input (np.ndarray): input current, same shape as t
            n_per_bin: binsize for firing rate computing
            max_iter: maximum iterations for the algorithm
        """
        n_iter = 0
        obj_rates = firing_rate(t, obj_spikes, n_per_bin)
        tweak_keys = ['tau_m', 'g_L']
        tweak_units = [ms, nS]
        init_pars = []
        for key, unit in zip(tweak_keys, tweak_units):
            init_pars.append(self.__getattribute__(key) / unit)
        init_pars = np.array(init_pars)

        def residuals(pars) -> np.ndarray:
            for key, par, unit in zip(tweak_keys, pars, tweak_units):
                self.__setattr__(key, par * unit)

            _, sim_spikes = self.simulate_trajectory(t, I_input)
            sim_rates = firing_rate(t, sim_spikes, n_per_bin)
            rate_error = (obj_rates - sim_rates)**2
            timing_error = sum([abs(t1 - t2) for t1, t2 in zip(sim_spikes, obj_spikes)])

            return  rate_error #+ timing_error

        #    ['tau_m', 'g_L', 'V_th']
        lb = [0.1,    0.1]
        up = [np.inf, np.inf]

        res_opt = least_squares(residuals, init_pars, bounds=(lb, up))
        print(f"error: {res_opt.fun}")

def test_lif():
    lif = LIF_model()
    # definimos tiempo y corriente
    t_arr = np.linspace(0, 400*ms, 1000)
    i_func = np.vectorize(lambda t : 130*pA*(t>100*ms)*(t<300*ms))
    I_arr = i_func(t_arr)
    # simulamos
    _, spike_times = lif.simulate_trajectory(t_arr, I_arr, plot=True)