import math
import numpy as np
from flask.cli import F
from icecream import ic

from scipy.constants import elementary_charge
from .Constants import t_0, E_a, k_b_ev, beta_0, L_p, E_m, gamma_drift, DifussiveBehaviour


def recombination(simu_time: float, pos_x: int, E_field: float, temperature: float, grid_size):
    # TODO: he modificado el vaor por la cara para ver si siendo mas alto sale algo
    Oxigen_Ion_velocity = t_0 * grid_size * \
        math.exp(-E_m / (k_b_ev * temperature)) * \
        math.sinh((2*elementary_charge * E_field * gamma_drift) / (k_b_ev * temperature))

    senoh = 10000 * math.sinh((2*elementary_charge * E_field * gamma_drift) / (k_b_ev * temperature))

    Prob_in_equilibrio = (simu_time * t_0)*(math.exp(-E_a / (k_b_ev * temperature)))

    Funcion = DifussiveBehaviour(pos_x, Oxigen_Ion_velocity, simu_time, grid_size)
    exponencial = math.exp(-(simu_time * Oxigen_Ion_velocity) / L_p)

    Prob_recom = Prob_in_equilibrio * (beta_0 * exponencial * Funcion)

    return Prob_recom, Oxigen_Ion_velocity, Funcion, senoh
