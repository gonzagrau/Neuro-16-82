import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Callable, Tuple, List, Dict # para hacer type hinting
from scipy.optimize import least_squares
# Importamos las constantes de unidades
from .units import pV, pA, pS, Mohm
from .units import nV, nA, nS, ns
from .units import uV, uA, uS, us
from .units import mV, mA, mS, ms

class Adex_model(object):
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
        # Parameters can be set either by passing them as kwargs...
        for name, value in kwargs.items():
            if name not in Adex_model.VALID_KEYS:
                raise ValueError(f"{name} is not a valid attribute for this class")
            setattr(self, name, value)
        # ... or set by default
        for def_name, def_val in Adex_model.DEFAULT_PARS.items():
            if getattr(self, def_name) is None:
                setattr(self, def_name, def_val)

    
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
                            I_units: float=pA,
                            t_units: float=ms,
                            v_units: float=mV) -> Tuple[np.ndarray, List[int]]:
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
                X[0, i-1] = 0
                u_next = self.V_reset
                w_next += self.b
                spike_times.append(i-1)

            X[:, i] = np.array([u_next, w_next])

        if plot:
            t /= t_units
            V = X[0, :]/v_units

            fig, ax = plt.subplots(figsize=(10, 5), sharex=True)
            
            ax.plot(t, V, 'k')
            ax.set_xlabel('$t$')
            ax.set_ylabel('$V$')

            plt.show()

        return X, spike_times
        

    def fit_spikes(self, t: np.ndarray, 
                   obj_spikes: List[int] | np.ndarray, 
                   I_input: np.ndarray) -> None:
        """
        Tweaks some model parameters to fit some objective spike times
        Args:
            t (np.ndarray): time array
            obj_spikes (List[int] | np.ndarray): list of indeces where spikes happen
            I_input (np.ndarray): input current, same shape as t
        """
        tweak_keys = ['tau_m', 'a', 'tau_w', 'b', 'V_reset']
        tweak_units = [ms,     ns,   ms,      pA,  mV]
        init_pars = []
        for key, unit in zip(tweak_keys, tweak_units):
            init_pars.append(self.__getattribute__(key)/unit)
        init_pars = np.array(init_pars)

        def residuals(pars) -> np.ndarray:
            for key, par, unit in zip(tweak_keys, pars, tweak_units):
                self.__setattr__(key, par*unit)

            _, sim_spikes = self.simulate_trajectory(t, I_input)

            len_obj = len(obj_spikes)
            len_sim = len(sim_spikes)
            null_spikes_arr = np.zeros(max(len_obj, len_sim))
            obj_spikes_lmax = null_spikes_arr.copy()
            sim_spikes_lmax = null_spikes_arr.copy()
            obj_spikes_lmax[:len_obj] = obj_spikes
            sim_spikes_lmax[:len_sim] = sim_spikes

            return obj_spikes_lmax - sim_spikes_lmax
        
        #    [tau_m', 'a', 'tau_w', 'b', 'V_reset']
        lb = [0.1,   -100, 0.1,     0.01, -65]
        up = [np.inf, 100, np.inf,  100,   0]

        res_opt = least_squares(residuals, init_pars, bounds=(lb, up))


def test_adex():
    adex = Adex_model()
    t_arr = np.linspace(0, 300*ms, 1000)
    i_0 = 65*pA
    i_func = np.vectorize(lambda t: i_0*(t>50*ms))
    I_arr = i_func(t_arr)

    obj_spikes = np.array([189, 200, 214, 235, 286, 409, 532, 655, 778, 901])
    adex.fit_spikes(t_arr, obj_spikes, I_arr)
    adex.simulate_trajectory(t_arr, I_arr, plot=True)


if __name__ == '__main__':
    test_adex()
