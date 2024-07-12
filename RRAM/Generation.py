import math
import numpy as np
from icecream import ic

from .Constants import *
from scipy.constants import elementary_charge
from .Representate import RepresentateStateOpt


def initial_state(Eje_x: float, Eje_y: float, num_trampas: int):
    """
    Generate an initial state for a grid with given dimensions and number of traps.

    Parameters:
    - Eje_x (int): The number of rows in the grid.
    - Eje_y (int): The number of columns in the grid.
    - num_trampas (int): The number of traps to be randomly placed in the grid.

    Returns:
    - InitialState (numpy.ndarray): The initial state grid with traps randomly placed.

    """
    # Create a matrix of zeros with size Eje_x x Eje_y

    InitialState = np.zeros((Eje_x, Eje_y), dtype=int)
    # Generate random positions for the traps

    posiciones_unos = np.random.choice(Eje_x * Eje_y, num_trampas, replace=False)

    # Assign the value 1 to the selected positions
    for pos in posiciones_unos:
        fila, columna = divmod(pos, Eje_x)
        InitialState[fila, columna] = 1
    return InitialState


def initial_state_priv(Eje_x: int, Eje_y: int, num_trampas: int, regiones_pesos: list):
    """
    Generate an initial state matrix with traps based on the given parameters with weighted regions.

    Args:
        Eje_x (int): The size of the x-axis.
        Eje_y (int): The size of the y-axis.
        num_trampas (int): The number of traps to generate.
        regiones_pesos (list): A list of tuples defining regions and their weights.
                               Each tuple should be ((x_start, x_end, y_start, y_end), weight).

    Returns:
        np.ndarray: The initial state matrix with traps.
    """
    # Create a matrix of zeros with size Eje_x x Eje_y
    InitialState = np.zeros((Eje_x, Eje_y), dtype=int)

    # Create a weight matrix initialized to 1
    pesos = np.ones((Eje_x, Eje_y), dtype=float)

    # Apply weights to specified regions
    for (x_start, x_end, y_start, y_end), weight in regiones_pesos:
        pesos[x_start:x_end, y_start:y_end] = weight

    # Flatten the weight matrix for use with np.random.choice
    pesos_flat = pesos.flatten()

    # Generate random positions for the traps with weights
    posiciones_unos = np.random.choice(Eje_x * Eje_y, num_trampas, replace=False, p=pesos_flat/np.sum(pesos_flat))

    # Assign the value 1 to the selected positions
    for pos in posiciones_unos:
        fila, columna = divmod(pos, Eje_y)
        InitialState[fila, columna] = 1

    return InitialState


def generation(paso_temporal: float, electric_field: float,
               temperature: float, carga_vacante: float = 2) -> float:
    """
    Calculate the generation rate of charge carriers in a RRAM device.

    Parameters:
    - paso_temporal (float): The time step for the simulation.
    - electric_field (float): The electric field applied to the device.
    - temperature (float): The temperature of the device.
    - carga_vacante (float, optional): The vacancy charge. Default is 2.

    Returns:
    - float: The generation rate of charge carriers.

    """
    exponente = (E_a - (gamma * carga_vacante * np.abs(electric_field))) / (k_b_ev * temperature)
    prob_generacion = paso_temporal * t_0 * (np.exp(-exponente))

    return prob_generacion
