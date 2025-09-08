from numpy.typing import NDArray
import numpy as np
import math

from scipy.constants import elementary_charge, Boltzmann, epsilon_0
from RRAM import Constants as cte


def Generate_Resitence_Matrix(configuration_matrix: NDArray, paths: list) -> NDArray:
    """
    Generates a resistance matrix based on the given configuration matrix and percolation paths.
    Args:
        configuration_matrix (np.ndarray): The initial configuration matrix.
        paths (list): A list of percolation paths, where each path is a list of (x, y) tuples representing coordinates.
    Returns:
        np.ndarray: A matrix of the same size as the configuration matrix, with positions marked as 1 where percolation paths exist.
    """
    # Crear una matriz de ceros del mismo tamaño que la configuración inicial
    percolation_matrix = np.zeros_like(configuration_matrix)

    # Iterar sobre cada camino de percolación y marcar las posiciones en la matriz
    for path in paths:
        for x, y in path:
            percolation_matrix[y, x] = 1

    return percolation_matrix


def OmhCurrent(
    potential: float, config_state: NDArray, **kwargs
) -> tuple[float, float]:
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
        ohm_resistence = float(kwargs.get("ohm_resistence"))  # pyright: ignore[reportArgumentType]
    else:
        ohm_resistence = 1.5

    # Initialize total resistance
    total_resistance = 0
    parallel_resistance = 0

    # Sobre cada columna de la matriz
    for row in config_state.T:
        # Se calcula la resistencia paralela de los elementos de la columna
        num_resistence = sum(row)
        parallel_resistance = 1 / (num_resistence / ohm_resistence)

        # for i in range(num_resistence):
        #     parallel_resistance += 1 / ohm_resistence

        # Se suma la resistencia paralela a la resistencia total
        total_resistance += parallel_resistance

        # Se calcula la corriente Ohmica
    return potential / total_resistance, total_resistance  # type: ignore


def Poole_Frenkel(temperature: float, E_field: float, **kwargs) -> float:
    """
    Calculate the current using the Poole-Frenkel effect.
    The Poole-Frenkel effect describes the lowering of the potential barrier in an insulator due to an applied electric field, which increases the current.
    Parameters:
    temperature (float): The temperature in Kelvin.
    E_field (float): The electric field in V/m.
    **kwargs: Optional keyword arguments for constants:
        - pb_metal_insul (float): Potential barrier between metal and insulator.
        - permitividad_relativa (float): Relative permittivity of the material.
        - I_0 (float): Pre-exponential factor.
    Returns:
    float: The current calculated using the Poole-Frenkel effect.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función hay q mejorarlo por si algun elemento fuera None
        potential_barrier = float(kwargs.get("pb_metal_insul"))  # type: ignore
        epsilon_r = float(kwargs.get("permitividad_relativa"))  # type: ignore
        I_0 = float(kwargs.get("I_0"))  # type: ignore
    else:
        potential_barrier = cte.pb_metal_insul
        epsilon_r = cte.permitividad_relativa
        I_0 = cte.I_0

    k_b_ev = Boltzmann / elementary_charge

    beta = math.sqrt(elementary_charge / (epsilon_0 * epsilon_r * math.pi))

    exponencial = math.exp(
        elementary_charge
        * (beta * math.sqrt(E_field) - potential_barrier)
        / (k_b_ev * temperature)
    )

    I_poole_frenkel = I_0 * E_field * exponencial

    return I_poole_frenkel
