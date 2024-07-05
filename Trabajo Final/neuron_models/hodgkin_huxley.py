import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict # para hacer type hinting
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from .utils import firing_rate, ruku4
# Importamos las constantes de unidades
from .utils import pV, pA, pS, Mohm
from .utils import nV, nA, nS, ns
from .utils import uV, uA, uS, us, uF
from .utils import mV, mA, mS, ms


class HodgkinHuxley(object):
    DEFAULT_PARS = {
            'g_Na' : 120.0*mS,
            'g_K' : 36.0*mS,
            'g_L' : 0.3*mS,
            'V_Na' : 50.0*mV,
            'V_K' : -77.0*mV,
            'V_L' : -54.4*mV,
            'C' : 1.0*uF,
            'V_init' : -65*mV,
            'n_init' : 0.318600,
            'm_init' : 0.052386,
            'h_init' : 0.592684,
    }
    VALID_KEYS = DEFAULT_PARS.keys()

    def __init__(self, **kwargs):
        self.g_Na = None
        self.g_K = None
        self.g_L = None
        self.V_Na = None
        self.V_K = None
        self.V_L = None
        self.C = None
        self.V_init = None
        self.n_init = None
        self.m_init = None
        self.h_init = None
        # Parameters can be set either by passing them as kwargs...
        for name, value in kwargs.items():
            if name not in HodgkinHuxley.VALID_KEYS:
                raise ValueError(f"{name} is not a valid attribute for this class")
            setattr(self, name, value)
        # ... or set by default
        for def_name, def_val in HodgkinHuxley.DEFAULT_PARS.items():
            if getattr(self, def_name) is None:
                setattr(self, def_name, def_val)


    @staticmethod
    def alpha_n(v: float):
        return 0.010*(v + 55*mV)/(1 - np.exp(-(v + 55*mV)/(10*mV)))

    @staticmethod
    def alpha_m(v: float) -> float:
        return 0.100 * (v + 40*mV) / (1 - np.exp(-(v + 40*mV) / (10*mV)))

    @staticmethod
    def alpha_h(v: float) -> float:
        return 0.070 * np.exp(-(v + 65*mV) / (20*mV))

    @staticmethod
    def beta_n(v: float) -> float:
        return 0.125 * np.exp(-(v + 65*mV) / 80*mV)

    @staticmethod
    def beta_m(v: float) -> float:
        return 4.0 * np.exp(-(v + 65*mV) / 18*mV)

    @staticmethod
    def beta_h(v: float) -> float:
        return 1 / (1 + np.exp(-(v + 35*mV) / 10*mV))


    def derivative(self, t: float, X: np.ndarray, i: Callable) -> np.ndarray:
        """
        Instantenous derivative for the Hodgkin and Huxley model
        :param: X: np.array([v, n, m, h])
        :param: i: current as a function of time
        :return: np.array([dv, dn, dm, dh])/dt
        """
        # Define constants
        s = self
        v = X[0]
        n = X[1]
        m = X[2]
        h = X[3]

        # Compute derivative
        dv = (i(t) - s.g_Na * m ** 3 * h * (v - s.V_Na) - s.g_K * n ** 4 * (v - s.V_K) - s.g_L * (v - s.V_L)) / s.C
        dn = s.alpha_n(v) * (1 - n) - s.beta_n(v) * n
        dm = s.alpha_m(v) * (1 - m) - s.beta_m(v) * m
        dh = s.alpha_h(v) * (1 - h) - s.beta_h(v) * h

        return np.array([dv, dn, dm, dh])


    def simulate_trajectory(self, t: np.ndarray,
                            i: Callable,
                            plot: bool = False,
                            t_units: float = ms,
                            v_units: float = mV) -> Tuple[np.ndarray, List[int]]:
        """
        :param np.ndarray t: time array of len N, defined as the range a:h:b
        :param np.ndarray i: current as a function of time
        :param plot: indicates whether to plot
        :param t_units: for plotting purposes
        :param v_units: for plotting purposes

        Uses the Runge-Kutta 4 method to solve the model

        :return  X such that. X[0, :] = V, X[1, :] = n, X[2, :] = m, X[3, :] = h, and spike times
        """
        s = self
        X_0 = np.array([s.V_init, s.n_init, s.m_init, s.h_init])
        X = ruku4(t, s.derivative, X_0, i=i)
        X = X.T
        V = X[0, :]
        spike_times = list(find_peaks(V, height=0)[0])

        if plot:
            fig, ax = plt.subplots()
            ax.plot(t/t_units, V/v_units, 'k-', linewidth=0.5)
            ax.set_xlabel('t [ms]')
            ax.set_ylabel('V [mV]')
            plt.show()

        return X, spike_times


def test_hyh():
    hyh_neuron = HodgkinHuxley()
    t = np.linspace(0, 100*ms, 100)
    i_0 = 10*uA  # [uA/cm^3]
    i = lambda t: i_0 * (t > 15*ms) * (t < 80*ms)

    hyh_neuron.simulate_trajectory(t, i, plot=True)


if __name__ == '__main__':
    test_hyh()
