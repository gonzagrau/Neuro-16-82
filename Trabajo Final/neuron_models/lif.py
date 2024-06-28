import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Callable, Tuple, List, Dict # para hacer type hinting
from scipy.optimize import least_squares
from .utils import firing_rate
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
                            I_units: float = pA,
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
            fig, ax = plt.subplots()
            ax.plot(t/t_units, V/v_units, 'k-', linewidth=0.5)
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('V [mV]')
            plt.show()

        return V, spike_times


    def fit_spikes(self, t: np.ndarray,
                   obj_spikes: List[int] | np.ndarray,
                   I_input: np.ndarray,
                   n_per_bin: int=10) -> None:
        """
        Tweaks some model parameters to fit some objective spike times
        Args:
            t (np.ndarray): time array
            obj_spikes (List[int] | np.ndarray): list of indeces where spikes happen
            I_input (np.ndarray): input current, same shape as t
            n_per_bin: binsize for firing rate computing
        """
        obj_rates = firing_rate(t, obj_spikes, n_per_bin)
        tweak_keys = ['tau_m', 'g_L', 'V_th']
        tweak_units = [ms, nS, mV]
        init_pars = []
        for key, unit in zip(tweak_keys, tweak_units):
            init_pars.append(self.__getattribute__(key) / unit)
        init_pars = np.array(init_pars)

        def residuals(pars) -> np.ndarray:
            for key, par, unit in zip(tweak_keys, pars, tweak_units):
                self.__setattr__(key, par * unit)

            _, sim_spikes = self.simulate_trajectory(t, I_input)
            sim_rates = firing_rate(t, sim_spikes, n_per_bin)

            return (obj_rates - sim_rates) + (len(obj_spikes) - len(sim_spikes))**2

        #    ['tau_m', 'g_L', 'V_th']
        lb = [0.1,    0.1,  -65]
        up = [np.inf, np.inf, 0]

        res_opt = least_squares(residuals, init_pars, bounds=(lb, up))


def test_lif():
    lif = LIF_model()
    # definimos tiempo y corriente
    t_arr = np.linspace(0, 400*ms, 1000)
    i_func = np.vectorize(lambda t : 130*pA*(t>100*ms)*(t<300*ms))
    I_arr = i_func(t_arr)
    # simulamos
    _, spike_times = lif.simulate_trajectory(t_arr, I_arr, plot=True)