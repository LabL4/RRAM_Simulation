import numpy as np


def SimpleElectricField(DDP: float, espesor: float) -> float:
    """
    Calcula el campo eléctrico en función de la diferencia de potencial y el espesor de la capa de óxido.
    Parameters:
    - DDP (float): Diferencia de potencial.
    - espesor (float): Espesor de la capa de óxido.
    Returns:
    - float: Campo eléctrico.
    """

    ElectricField = abs(DDP / espesor)

    return ElectricField


def GapElectricField(
    potential: float, pos_y: int, actual_state: np.ndarray, device_size_x: float, grid_size: float
) -> float:
    """
    Calculate the non-uniform electric field in the device along the X axis (electrode direction).
    This function computes the electric field based on the potential difference,
    the row position (Y index), and the current state of the device. The physical
    distance between electrodes (device_size_x) is used as the reference length.

    Parameters:
        - potential (float): The potential difference applied across the device.
        - pos_y (int): The row index (Y position) for which to evaluate the gap along X.
        - actual_state (np.ndarray): The current state of the device as a 2D array.
        - device_size_x (float): The physical distance between electrodes (X axis) in meters.
        - grid_size (float): The physical size of one cell (atom_size) in meters.

    Returns:
        - float: The electric field across the remaining gap in the X direction.
    """

    gap = grid_size * (np.sum(actual_state[pos_y]))
    L = device_size_x - gap
    if L == 0:
        return potential / device_size_x
    else:
        return potential / L
