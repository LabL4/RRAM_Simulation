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

    ElectricField = DDP / espesor

    return ElectricField


def GapElectricField(potential: float, pos_y: int, actual_state: np.array, **kwargs) -> float:
    """
    Calculate the No-normal electric field the device.
    This function computes the electric field based on the potential difference,
    the position in the y-axis, and the current state of the device. The size of 
    the device and the atomic size can be passed as keyword arguments; otherwise, 
    default values are used.

    Parameters:
        - potential (float): The potential difference applied across the device.
        - pos_y (int): The y-axis position to evaluate the gap.
         -actual_state (np.array): The current state of the device, represented as a 2D array.

    **kwargs: Optional keyword arguments for:
        - 'device_size' (float): The size of the device in meters. Default is 10e-9.
        - 'atom_size' (float): The size of an atom in meters. Default is 0.25e-9.

    Returns:
        - float: The electric field across the gap in the device.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        size_device = float(kwargs.get('device_size'))
        cte_red = float(kwargs.get('atom_size'))
    else:
        size_device = 10e-9
        cte_red = 0.25e-9

    # Como solo tengo 0 y 1 la cantidad de 1 es directamente la suma de la fila
    gap = cte_red*(np.sum(actual_state[pos_y]))

    L = size_device - gap

    E_field = potential / L

    return E_field
