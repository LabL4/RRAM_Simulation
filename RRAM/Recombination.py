import math
import numpy as np

from scipy.constants import elementary_charge
from .Constants import t_0, E_a, k_b_ev, beta_0, L_p, E_m, gamma_drift, DifussiveBehaviour


def recombination(simu_time: float, pos_x: int, E_field: float, temperature: float, grid_size, factor: float):

    Prob_in_equilibrio = (simu_time * t_0)*(math.exp(-E_a / (k_b_ev * temperature)))

    senoh = math.sinh((2*elementary_charge * E_field * pos_x * gamma_drift) / (k_b_ev * temperature))
    exponencial_velocidad = math.exp(-E_m / (k_b_ev * temperature))

    # TODO: he modificado el valor por la cara para ver si siendo mas alto sale algo
    Oxigen_Ion_velocity = factor * t_0 * grid_size * (senoh * exponencial_velocidad)

    Funcion = DifussiveBehaviour(pos_x, Oxigen_Ion_velocity, simu_time, grid_size)
    exponencial = math.exp(-(simu_time * Oxigen_Ion_velocity) / L_p)
    beta = (beta_0 * exponencial * Funcion)

    Prob_recom = Prob_in_equilibrio * beta

    return Prob_recom, Oxigen_Ion_velocity*simu_time, Funcion
