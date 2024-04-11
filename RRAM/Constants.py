# Módulo que contiene todas las constantes que usa el paquete

import numpy as np
from scipy.constants import Boltzmann, elementary_charge

# Hz,  el valor de 1/t_0
t_0 = 10**13

# eV
E_a = 1

# Tiene distintos valores dependiendo si el estado es FORMING/SET y RESET
gamma = 3

# Valor de la constante de Boltzmann en eV/K
k_b_ev = Boltzmann / elementary_charge

# Factor de recombinación debido a la presencia de exceso de iones de oxígeno
beta_0 = 3e3

# m, longitud de decaimiento de la concentración de oxígeno
L_p = 1e-9

# Coeficiente de deriva de los iones de oxígeno debido a la presencia de un campo eléctrico externo
gamma_drift = 8

# Potencial de migración de los iones de oxígeno en eV
E_m = 1.0

# Constante de resistencia térmica en K/W
resistencia_termica = 5e5

# Constante de resistencia térmica en ohmios
resistencia_ohmica = 1e6


def DifussiveBehaviour(
        pos_x: int, Oxigen_Ion_velocity: float, Simulation_time: float,
        grid_size: float = 0.25e-9) -> float:
    """
    Calculates the diffusion behavior based on the given parameters.

    Parameters:
    - pos_x (int): The position of the diffusion event.
    - Oxigen_Ion_velocity (float): The velocity of the oxygen ion.
    - Simulation_time (float): The simulation time.
    - grid_size (float, optional): The size of the grid. Default is 0.25e-9.

    Returns:
    - float: The diffusion value based on the given conditions.

    """

    vt = Oxigen_Ion_velocity*Simulation_time
    pos_x = pos_x*grid_size

    condiciones = [pos_x <= vt,
                   (vt < pos_x) and (pos_x <= vt + grid_size),
                   (vt + grid_size < pos_x) and (pos_x <= vt + 3*grid_size),
                   (pos_x > vt + 3*grid_size)
                   ]

    valores_Difusion = [1, 0.3, 0.1, 0]
    return np.piecewise(pos_x, condiciones, valores_Difusion)
