import numpy as np
from typing import Tuple, List # para hacer type hinting
# Importaciones de la misma libreria
from .base_model import NeuronModel
from .utils import plot_voltage
# Importamos las constantes de unidades
from .utils import  pA, Mohm
from .utils import nS, ns
from .utils import mV, mS, ms


class Adex_model(NeuronModel):
    DEFAULT_PARS = {
        'tau_m': 5*ms,
        'R': 500*Mohm,
        'V_rest': -70*mV,
        'V_reset': -51*mV,
        'V_rh': -50.0*mV,
        'delta_T': 2.0*mV,
        'a': 0.5*nS,
        'tau_w': 100*mS,
        'b': 7.0*pA,
        'V_init': -70.0*mV,
        'w_init': 0*mV,
        'V_thres': -35*mV,
        'V_postreset': 0 * mV
    }
    VALID_KEYS = DEFAULT_PARS.keys()

    def __init__(self, **kwargs):
        self.tau_m = None
        self.R = None
        self.V_rest = None
        self.V_reset = None
        self.V_rh = None
        self.delta_T = None
        self.a = None
        self.tau_w = None
        self.b = None
        self.V_init = None
        self.w_init = None
        self.V_thres = None
        self.V_postreset = None
        # Parameters can be set either by passing them as kwargs...
        super().__init__(Adex_model.DEFAULT_PARS, Adex_model.VALID_KEYS, **kwargs)


    def derivative(self, I_val: float, u: float, w: float) -> np.ndarray:
        """
        Computes instantaneous derivative
        Args:
            I_val (float): instantaneous current
            u (float): instantaneous voltage
            w (float): instantenous adaptative current

        Returns:
            np.ndarray: np.array([du/dy, dv/dt])
        """
        s = self # alias for neater code

        du = -(u - s.V_rest) + s.delta_T*np.exp((u - s.V_rh)/s.delta_T) - s.R*w + s.R*I_val
        dw = s.a*(u - s.V_rest) - w

        return np.array([du/s.tau_m, dw/s.tau_w])


    def simulate_trajectory(self, t: np.ndarray,
                            I_input: np.ndarray,
                            plot: bool=False,
                            t_units: float=ms,
                            v_units: float=mV) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resuelve numericamente un problema de valor inicial para el modelo LIF

        t: muestras temporales equiespaciadas
        I_arr: muestras de la corriente en los instantes de t
        plot: indica si graficar o no la trayectoria junto al estímulo
        I_units: defaultea a [pA]
        t_units: defaultea a [ms]
        v_units: defaultea a [mV]

        returns: X, tal que X[0, :] = V, y X[1, :] = w
                spike_times: lista con los indices donde ocurre un disparo

        """
        X_0 = np.array([self.V_init, self.w_init])

        t_len = len(t)
        dt = t[1] - t[0]
        spike_times = []

        # inicializamos V
        X = np.zeros((2, t_len)) # X[0, :] = u ; X[1, :] = w
        X[:, 0] = X_0

        # Simulamos
        for i in range(1, t_len):
            X_prev = X[:, i-1]
            X_next = X_prev + self.derivative(I_input[i-1], X_prev[0], X_prev[1])*dt
            u_next, w_next = X_next[0], X_next[1]

            # Caso: disparo, y tenemos que resetear
            if u_next > self.V_thres:
                X[0, i-1] = self.V_postreset
                u_next = self.V_reset
                w_next += self.b
                spike_times.append(i-1)

            X[:, i] = np.array([u_next, w_next])

        if plot:
            V = X[0, :]
            plot_voltage(t, V, t_units, v_units)

        return X, np.array(spike_times)


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
                tweak_keys = ['tau_m', 'a', 'tau_w', 'b', 'V_reset', 'delta_T']
                tweak_units = [ms, ns, ms, pA, mV, mV]
            else:
                tweak_units = [1 for _ in range(len(tweak_keys))]

        super().fit_spikes(t, obj_spikes, I_input, n_per_bin, tweak_keys,
                           tweak_units, N_iter, max_rep, pop_size, mut_rate)


def test_adex():
    # 1. Runs simulations with a series of known parameters
    t_arr = np.linspace(0, 300 * ms, 1000)
    i_0 = 65 * pA
    i_func = np.vectorize(lambda t: i_0 * (t > 50 * ms))
    I_arr = i_func(t_arr)

    pars_adex_init_burst = {'V_thres' : 0,
                            'tau_m' : 5*ms,
                            'a' : 0.5*nS,
                            'tau_w' : 100*ms,
                            'b' : 7*pA,
                            'V_reset' : -51*mV}
    pars_adex_bursting = {'V_thres' : 0,
                          'tau_m' : 5*ms,
                          'a' : -0.5*nS,
                          'tau_w' : 100*ms,
                          'b' : 7*pA,
                          'V_reset' : -46*mV}
    patterns = [pars_adex_bursting, pars_adex_init_burst]
    X_list = []
    spikes_list = []
    for kwargs in patterns:
        adex = Adex_model(**kwargs)
        X, spike_times = adex.simulate_trajectory(t_arr, I_arr, plot=True)
        X_list.append(X)
        spikes_list.append(spike_times)

    # 2. Tries to fit the parameters to match
    base_adex = Adex_model()
    obj_spikes = spikes_list[0]
    base_adex.fit_spikes(t_arr, obj_spikes, I_arr, n_per_bin=len(t_arr)//30)
    base_adex.simulate_trajectory(t_arr, I_arr, plot=True)