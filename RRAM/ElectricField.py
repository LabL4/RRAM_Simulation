import numpy as np
import logging

logger = logging.getLogger(__name__)


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
        return abs(potential / L)


def GapElectricFieldVector(
    potential: float, actual_state: np.ndarray, device_size_x: float, grid_size: float
) -> np.ndarray:
    """
    Campo eléctrico local de TODAS las filas a la vez (versión vectorizada de `GapElectricField`).

    Devuelve un vector (Ny,) con, para cada fila, `|potential / (device_size_x - gap)|`, donde `gap`
    es la longitud ocupada por vacantes en esa fila. Si la fila está tan llena que el hueco se anula
    (L == 0) devuelve `potential / device_size_x`, igual que la versión escalar. Mismos números,
    bit a bit, con una sola llamada de NumPy en lugar de un bucle de Python por fila.
    """
    gap = grid_size * np.sum(actual_state, axis=1)
    L = device_size_x - gap
    anulado = L == 0
    L_seguro = np.where(anulado, 1.0, L)  # evita la división por cero en las filas anuladas
    return np.where(anulado, potential / device_size_x, np.abs(potential / L_seguro))
