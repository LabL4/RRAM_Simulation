import math
import numpy as np

from scipy.constants import elementary_charge
from .Constants import t_0, E_a, k_b_ev, beta_0, L_p, E_m, gamma_drift, DifussiveBehaviour


def recombination(
        simulation_time: float, pos_x: float, electric_field: float, temperature: float, grid_size=0.25e-9):

    Oxigen_Ion_velocity = beta_0 * grid_size * \
        math.exp(-E_m / (k_b_ev * temperature)) * \
        math.sinh((2*elementary_charge * grid_size * electric_field * gamma_drift) /
                  (k_b_ev * temperature))

    Prob_in_equilibrio = (simulation_time * t_0) / \
        (math.exp(-E_a /
         (k_b_ev * temperature)))

    Recombination_Probability = beta_0(
        math.exp(-simulation_time * Oxigen_Ion_velocity / L_p)) * \
        DifussiveBehaviour(pos_x, Oxigen_Ion_velocity, simulation_time, grid_size)

    return Prob_in_equilibrio * Recombination_Probability
