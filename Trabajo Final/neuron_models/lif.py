import numpy as np
from typing import Tuple, List # para hacer type hinting
# Importaciones de la misma libreria
from .base_model import NeuronModel
from .utils import plot_voltage
# Importamos las constantes de unidades
from .utils import pA
from .utils import nS
from .utils import mV, ms


class LIF_model(NeuronModel):
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
        super().__init__(LIF_model.DEFAULT_PARS, LIF_model.VALID_KEYS, **kwargs)


    def derivative(self, I_val: float, u: float) -> np.ndarray:
        """
        Computes the instantaneous derivative
        :param I_val: current current (hehe)
        :param u: current voltage
        :return: du/dt
        """
        s = self  # alias for neat code
        du = -(u - s.E_L) + I_val / s.g_L
        return du / s.tau_m


    def simulate_trajectory(self, t: np.ndarray,
                            I_input: np.ndarray,
                            plot: bool = False,
                            t_units: float = ms,
                            v_units: float = mV) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves IVP to find the trajectory v(t)
        :param t: time samples
        :param I_input: input current, same shape as t
        :param plot: indicates whether to plot
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

        return V, np.array(spike_times)


    def fit_spikes(self, t: np.ndarray,
                  obj_spikes: np.ndarray,
                  I_input: np.ndarray,
                  n_per_bin: int=10,
                  tweak_keys: List['str'] | None=None,
                  tweak_units: List[int | float] | None=None,
                  N_iter: int=1000,
                  max_rep: int=10,
                  pop_size: int=100,
                  mut_rate: float=0.01,
                  mut_scale: float=1.) -> None:
        """
        See docstring for superclass
        """
        if tweak_units is None:
            if tweak_keys is None:
                tweak_units = [ms, nS]
                tweak_keys = ['tau_m', 'g_L']
            else:
                tweak_units = [1 for _ in range(len(tweak_keys))]

        super().fit_spikes(t, obj_spikes, I_input, n_per_bin, tweak_keys,
                           tweak_units, N_iter, max_rep, pop_size, mut_rate)


def test_lif():
    lif = LIF_model()
    # definimos tiempo y corriente
    t_arr = np.linspace(0, 400*ms, 1000)
    i_func = np.vectorize(lambda t : 130*pA*(t>100*ms)*(t<300*ms))
    I_arr = i_func(t_arr)
    # simulamos
    _, spike_times = lif.simulate_trajectory(t_arr, I_arr, plot=True)