import numpy as np

from RRAM import Constants as cte

k_b_ev = 8.617333262145e-5  # Boltzmann constant in eV/K


def vecinos_verticales(matriz, i, j):
    x_size = matriz.shape[0]
    return (i > 0 and matriz[i - 1, j] > 0) or (i < x_size - 1 and matriz[i + 1, j] > 0)


def vecinos_horizontales(matriz, i, j):
    y_size = matriz.shape[1]
    return (j > 0 and matriz[i, j - 1] > 0) or (j < y_size - 1 and matriz[i, j + 1] > 0)


def vecinos_izquierda(matriz, i, j):
    """
    Comprueba si el elemento en la posición (i, j) tiene un elemento a su izquierda (j-1) mayor que 0.
    """
    return j > 0 and matriz[i, j - 1] > 0


def tiene_vecinos(matriz, i, j):
    x_size, y_size = matriz.shape
    return (
        (i > 0 and matriz[i - 1, j] > 0)  # arriba
        or (i < x_size - 1 and matriz[i + 1, j] > 0)  # abajo
        or (j > 0 and matriz[i, j - 1] > 0)  # izquierda
        or (j < y_size - 1 and matriz[i, j + 1] > 0)  # derecha
    )


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

    InitialState = np.zeros((Eje_x, Eje_y), dtype=int)  # type: ignore
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
    posiciones_unos = np.random.choice(Eje_x * Eje_y, num_trampas, replace=False, p=pesos_flat / np.sum(pesos_flat))

    # Assign the value 1 to the selected positions
    for pos in posiciones_unos:
        fila, columna = divmod(pos, Eje_y)
        InitialState[fila, columna] = 1

    return InitialState


def Generate(
    time_stp: float,
    electric_field: float,
    temp: float,
    vibration_frequency: float,
    generation_energy: float,
    cte_red: float,
    gamma: float,
) -> float:
    """
    Calculates the generation probability of RRAM devices.
    Args:
        time_stp (float): The time step for the calculation.
        electric_field (float): The electric field applied to the device.
        temp (float): The temperature of the device.
        **kwargs: Contains the constants needed for the calculation.
    Keyword Args:
        vibration_frequency (float): The vibration frequency constant. Required if kwargs is provided.
        activation_energy (float): The activation energy constant. Required if kwargs is provided.
        cte_red (float): The reduction constant. Required if kwargs is provided.
        gamma (float): The gamma constant. Required if kwargs is provided.
    Returns:
        float: The generation probability of generate a vancancy.
    """

    exponente = (generation_energy - (gamma * cte_red * electric_field)) / (k_b_ev * temp)
    prob_generacion = time_stp * vibration_frequency * (np.exp(-exponente))

    return prob_generacion


def Generate_vectorized(
    time_stp: float,
    electric_field_matrix: np.ndarray,
    temp: np.ndarray,
    vibration_frequency: float,
    generation_energy: float,
    cte_red: float,
    gamma: float,
) -> np.ndarray:
    """
    Calcula la matriz de probabilidades de generación para una matriz de campo eléctrico.

    Parámetros:
        time_stp (float): Paso temporal.
        electric_field_matrix (np.ndarray): Matriz 2D con valores del campo eléctrico, cada fila tiene el mimso valor de campo eléctrico.
        temp (float): Temperatura (escala única para toda la matriz).
        **kwargs: Constantes necesarias (vibration_frequency, activation_energy, cte_red, gamma).

    Retorna:
        np.ndarray: Matriz con probabilidades de generación (mismo tamaño que electric_field_matrix).
    """

    exponente = (generation_energy - (gamma * cte_red * electric_field_matrix)) / (k_b_ev * temp)
    prob_matrix = time_stp * vibration_frequency * np.exp(-exponente)

    # Limitar probabilidades máximas a 1
    prob_matrix = np.minimum(prob_matrix, 1.0)

    return prob_matrix
