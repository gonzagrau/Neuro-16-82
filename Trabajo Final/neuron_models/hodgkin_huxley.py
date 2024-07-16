import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict # para hacer type hinting
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from .base_model import NeuronModel
from .utils import firing_rate, ruku4, plot_voltage
# Importamos las constantes de unidades
from .utils import pV, pA, pS, Mohm
from .utils import nV, nA, nS, ns
from .utils import uV, uA, uS, us, uF
from .utils import mV, mA, mS, ms, cm


class HodgkinHuxley(NeuronModel):
    DEFAULT_PARS = {
            'g_Na' : 120.0,
            'g_K' : 36.0,
            'g_L' : 0.3,
            'V_Na' : 50.0,
            'V_K' : -77.0,
            'V_L' : -54.4,
            'C' : 1.0,
            'V_init' : -65.,
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

        super().__init__(HodgkinHuxley.DEFAULT_PARS, HodgkinHuxley.VALID_KEYS, **kwargs)

    @staticmethod
    def alpha_n(v: float):
        return 0.010 * (v + 55)/(1 - np.exp(-(v + 55)/(10)))

    @staticmethod
    def alpha_m(v: float) -> float:
        return 0.100 * (v + 40) / (1 - np.exp(-(v + 40) / (10)))

    @staticmethod
    def alpha_h(v: float) -> float:
        return 0.070 * np.exp(-(v + 65) / (20))

    @staticmethod
    def beta_n(v: float) -> float:
        return 0.125 * np.exp(-(v + 65) / 80)

    @staticmethod
    def beta_m(v: float) -> float:
        return 4.0 * np.exp(-(v + 65) / 18)

    @staticmethod
    def beta_h(v: float) -> float:
        return 1 / (1 + np.exp(-(v + 35) / 10))


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
                            t_units: float = 1,
                            v_units: float = 1,
                            title: str | None = None) -> Tuple[np.ndarray, List[int]]:
        """
        :param np.ndarray t: time array of len N, defined as the range a:h:b
        :param np.ndarray i: current as a function of time
        :param plot: indicates whether to plot
        :param t_units: for plotting purposes
        :param v_units: for plotting purposes
        :param title: for plotting purposes

        Uses the Runge-Kutta 4 method to solve the model

        :return  X such that. X[0, :] = V, X[1, :] = n, X[2, :] = m, X[3, :] = h, and spike times
        """
        s = self
        X_0 = np.array([s.V_init, s.n_init, s.m_init, s.h_init])
        derivative = lambda t_val, X_arr, i_func : self.derivative(t_val, X_arr, i_func)
        res = solve_ivp(derivative, t_span=(t[0], t[-1]), args=(i,), t_eval=t, method='LSODA', y0=X_0)
        X = res.y
        V = X[0, :]
        spike_times = list(find_peaks(V, height=0)[0])

        if plot:
            plot_voltage(t, V, t_units, v_units, title)

        return X, spike_times


def test_hyh():
    hyh_neuron = HodgkinHuxley()
    t = np.linspace(0, 100, 1000)
    i_0 = 10  # [uA/cm^3]
    i = lambda t_i: i_0 * (t_i > 15) * (t_i < 80)

    hyh_neuron.simulate_trajectory(t, i, plot=True)


if __name__ == '__main__':
    test_hyh()
