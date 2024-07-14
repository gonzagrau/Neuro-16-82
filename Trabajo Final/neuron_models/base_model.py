import numpy as np
from typing import List, Dict, Self, Iterable # para hacer type hinting
from .genetic_algo import GeneticAlgorithm
from .utils import firing_rate, plot_voltage


class NeuronModel(object):
    """
    Base class for all neuron models
    Adex, LIF, and Hodgkin-Huxley are sublclasses of this class
    this defines how to initialize, get, and set parameters
    """
    def __init__(self, default_values: Dict[str, float], valid_keys: Iterable[str], **kwargs):
        # Parameters can be set either by passing them as kwargs...
        for name, value in kwargs.items():
            if name not in valid_keys:
                raise ValueError(f"{name} is not a valid attribute for this class")
            setattr(self, name, value)
        # ... or set by default
        for def_name, def_val in default_values.items():
            if getattr(self, def_name) is None:
                setattr(self, def_name, def_val)


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


    def fit_spikes(self, t: np.ndarray,
                   obj_spikes: np.ndarray,
                   I_input: np.ndarray,
                   n_per_bin: int=10,
                   tweak_keys: List['str'] | None=None,
                   tweak_units: List[int | float] | None=None,
                   N_iter: int=1000,
                   max_rep: int=10,
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
        :param max_rep: maximum amount of succesive unevolutive generations
        :param mut_rate: float in [0, 1) for mutation rate in GA
        :return: None, but the internal parameters are tweaked to the best fitting
        """

        obj_rates = firing_rate(t, obj_spikes, n_per_bin)
        init_pars = self.get_init_pars_2_fit(tweak_keys, tweak_units)

        def fitness_function(pars: np.ndarray, lif_obj: Self, keys, units):
            lif_obj.update_params(keys, pars, units)
            _, sim_spikes = lif_obj.simulate_trajectory(t, I_input)
            sim_rates = firing_rate(t, sim_spikes, n_per_bin)
            rate_error = np.sum((obj_rates - sim_rates)**2)
            # timing_error = sum([abs(t1 - t2) for t1, t2 in zip(sim_spikes, obj_spikes)])
            return 1 / (1 + rate_error)

        algo_obj = GeneticAlgorithm(pop_size, N_iter, max_rep, mut_rate, fitness_function, init_pars)
        best_solution = algo_obj.genetic_algorithm(self, tweak_keys, tweak_units)

        self.update_params(tweak_keys, best_solution, tweak_units)