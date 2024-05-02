import math
import numpy as np
from icecream import ic

from .Constants import *
from scipy.constants import elementary_charge


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


def generation(simulation_time: np.ndarray, electric_field: np.ndarray,
               temperature: np.ndarray, carga_vacante: float = 2,) -> np.ndarray:
    """
    Calculate the generation rate of RRAM devices.

    Parameters:
        - simulation_time (numpy.ndarray): Array of simulation time values.
        - electric_field (numpy.ndarray): Array of electric field values.
        - temperature (numpy.ndarray): Array of temperature values.
        - carga_vacante (float, optional): Vacancy charge value (default: 2).

    Returns:
    - numpy.ndarray
        Array of generation rates.

    """
    exponente = (E_a - (gamma * carga_vacante * np.abs(electric_field))) / (k_b_ev * temperature)

    return (simulation_time * t_0 * (np.exp(-exponente)))


if __name__ == "__main__":

    Longitud_Dispositivo = 10
    Atom_size = 0.5

    Eje_x = round(Longitud_Dispositivo / Atom_size)
    Eje_y = round(Longitud_Dispositivo / Atom_size)

    num_trampas = 10

    # Crear una matriz de ceros de tamaño Eje_x x Eje_y

    InitialState = np.zeros((Eje_x, Eje_y), dtype=int)
    # Generar 5 posiciones aleatorias para los unos

    posiciones_unos = np.random.choice(Eje_x * Eje_y, num_trampas, replace=False)

    # Asignar el valor 1 a las posiciones seleccionadas
