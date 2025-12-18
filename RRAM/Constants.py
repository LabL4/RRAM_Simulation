# Módulo que contiene todas las constantes que usa el paquete
# import numpy as np

from scipy.constants import Boltzmann, elementary_charge  # type: ignore

# Valor de la constante de Boltzmann en eV/K
k_b_ev = Boltzmann / elementary_charge

# Hz,  el valor de 1/t_0
t_0 = 1e13

# eV Energía de activación
E_a = 1  # puedo cambiar el valor

# eV Energía de activación recombinación
E_r = 1  # puedo cambiar el valor

# Tiene distintos valores dependiendo si el estado es FORMING/SET y RESET
gamma = 3

# Factor de recombinación debido a la presencia de exceso de iones de oxígeno
beta_0 = 3e3

# m, longitud de decaimiento de la concentración de oxígeno
L_p = 1e-9

# Coeficiente de deriva de los iones de oxígeno debido a la presencia de un campo eléctrico externo
gamma_drift = 10

# Potencial de migración de los iones de oxígeno en eV
E_m = 0.6

# Constante de resistencia térmica en K/W cuando el sistema no percola
r_termica_no_percola = 10  # bajar dos órdenes de magnitud

# Constante de resistencia térmica en K/W cuando el sistema percola
r_termica_percola = 500

# Constante de resistencia en ohmios
ohm_resistence = 350

# Constante de red, el paper original propone 0.25 nm
cte_red = 0.125e-9

# the potential barrier at the metal and insulator interface [eV]
pb_metal_insul = 0.2688
pb_metal_insul_reset = 0.2549

# Permitividad relativa del material
permitividad_relativa = 499.9631
permitividad_relativa_reset = 385.4334

# Término inicial de la ecuación de Poole-Frenkel
I_0 = 1.6501e-01
I_0_reset = 9.3536e-02

# Factor por el que divido la gamma cuando el sistema percola o no
factor_generacion = 1.5

# def DifussiveBehaviour(pos_x: int, oxigen_velocity: float, paso_temp: float, grid_size: float) -> float:
#     """
#     Calculates the diffusion behavior based on the given parameters.

#     Parameters:
#     - pos_x (int): The position of the diffusion event.
#     - Oxigen_Ion_velocity (float): The velocity of the oxygen ion.
#     - Simulation_time (float): The simulation time.
#     - grid_size (float, optional): The size of the grid. Default is 0.25e-9.

#     Returns:
#     - float: The diffusion value based on the given conditions.
#     """

#     vt = oxigen_velocity*paso_temp
#     pos_x = pos_x*grid_size

#     condiciones = [pos_x <= vt,
#                    (vt < pos_x) and (pos_x <= vt + grid_size),
#                    (vt + grid_size < pos_x) and (pos_x <= vt + 3 * grid_size),
#                    (pos_x > vt + 3 * grid_size)
#                    ]

#     valores_Difusion = [1, 0.3, 0.1, 0]
#     valor_elegido = np.piecewise(pos_x, condiciones, valores_Difusion).item()

#     return valor_elegido
