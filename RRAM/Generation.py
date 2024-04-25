import math
import numpy as np

from scipy.constants import elementary_charge
from .Constants import E_a, gamma, k_b_ev, t_0


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
    print(posiciones_unos)

    # Assign the value 1 to the selected positions
    for pos in posiciones_unos:
        fila, columna = divmod(pos, Eje_x)
        InitialState[fila, columna] = 1
    return InitialState


def generation(
        simulation_time: float, electric_field: float, temperature: float,
        grid_size: float = 0.25e-9, carga_vacante: float = 2) -> float:
    """
    Calculates the generation rate of a certain process.

    Parameters:
    - Delta_t (float): Time step.
    - electric_field (float): Electric field strength in V/m.
    - temperature (float): Temperature in Kelvin.
    - grid_size (float, optional): Size of the grid in meters. Default is 0.25e-9.
    - carga_vacante (int, optional): Vacancy charge. Default is 2.

    Returns:
    - generation_rate (float): The calculated generation rate.
    """

    # TODO: METERLE REESCALADO PARA INTENTAR ENVITAR TRATAR CON NUMEROS TAN GRANDES Y PEQUEÑOS.

    exponente = (E_a - (gamma * grid_size * carga_vacante * elementary_charge * abs(electric_field))) /\
                (k_b_ev * temperature)

    return simulation_time * t_0 * (math.exp(-exponente))


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
