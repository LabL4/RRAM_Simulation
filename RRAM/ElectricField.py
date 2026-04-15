import numpy as np


def SimpleElectricField(DDP: float, dist_electrodos: float) -> float:
    """
    Calcula el campo eléctrico en función de la diferencia de potencial y el espesor de la capa de óxido.
    Parameters:
    - DDP (float): Diferencia de potencial.
    - dist_electrodos (float): Espesor de la capa de óxido.
    Returns:
    - float: Campo eléctrico.
    """

    ElectricField = abs(DDP / dist_electrodos)

    return ElectricField


def GapElectricField(
    potential: float, pos_y: int, actual_state: np.ndarray, device_size_x: float, grid_size: float
) -> float:
    """
    Calcula el campo eléctrico local en la fila pos_y del dieléctrico.

    Parámetros:
        - potential (float): Diferencia de potencial aplicada [V].
        - pos_y (int): Índice de fila a evaluar (shape[0] de actual_state).
        - actual_state (np.ndarray): Estado actual del dieléctrico.
        - device_size_x (float): Distancia entre electrodos [m] → shape[0] * atom_size.
        - grid_size (float): Tamaño de celda de la malla [m].

    Retorna:
        - float: Campo eléctrico local [V/m].
    """
    gap = grid_size * (np.sum(actual_state[pos_y]))
    L = device_size_x - gap
    if L == 0:
        return potential / device_size_x
    else:
        return potential / L
