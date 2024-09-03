import math
import numpy as np

from RRAM import Constants as cte
from scipy.constants import elementary_charge, Boltzmann


def OmhCurrent(potential: float, config_state: np.array, **kwargs) -> float:
    """
    Calculates the Ohmic current based on the given parameters.

    Parameters:
    - DDP (float): The voltage difference across the circuit.
    - config_state (np.array): The configuration state of the circuit.
    - ohm_resistence (float): The default resistance value.

    Returns:
    - float: The calculated Ohmic current.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        ohm_resistence = float(kwargs.get('ohm_resistence'))
    else:
        ohm_resistence = cte.ohm_resistence

    # Initialize total resistance
    total_resistance = 0
    parallel_resistance = 0
    # Sobre cada columna de la matriz
    for column in config_state.T:
        # Comprobar si la columna es nula completamente
        if all(ohm_resistence == 0 for ohm_resistence in column):
            continue  # Saltar a la siguiente columna si es nula completamente

        # Se calcula la resistencia paralela de los elementos de la columna
        parallel_resistance = sum(1 / ohm_resistence for x in column if x == 1)

        # Se suma la resistencia paralela a la resistencia total
        total_resistance += 1 / parallel_resistance

    # Se calcula la corriente Ohmica
    return potential / total_resistance


def poole_frenkel(temperature: float, electric_field: float,
                  barrera: float = 0.9, beta: float = 1.697035E-5, I_0: float = 1e2) -> float:
    """
    Calculates the current using the Poole-Frenkel equation.

    Args:
        temperature (float): The temperature in Kelvin.
        electric_field (float): The electric field strength in V/m.
        barrera (float, optional): The barrier height in eV. Defaults to 0.895.
        beta (float, optional): The Poole-Frenkel constant. Defaults to 1.697035E-5.
        I_0 (float, optional): The saturation current in A. Defaults to 1e-12.

    Returns:
        float: The calculated current in A/m^2.
    """

    k_b_ev = Boltzmann / elementary_charge

    exponencial = np.exp(elementary_charge * (beta * np.sqrt(electric_field) - barrera) / (k_b_ev * temperature))

    I_poole_frenkel = electric_field * exponencial

    return I_poole_frenkel
