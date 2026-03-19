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


def calcular_probabilidad_generacion(
    time_stp: float,
    electric_field: np.ndarray | float,
    temp: np.ndarray | float,
    vibration_frequency: float,
    generation_energy: float,
    cte_red: float,
    gamma: float,
) -> np.ndarray | float:
    """
    Calcula la matriz de probabilidades de generación para una matriz de campo eléctrico.
    Soporta tanto escalares (float) como matrices (np.ndarray) para el campo y la temperatura.
    """

    exponente = (generation_energy - (gamma * cte_red * electric_field)) / (k_b_ev * temp)
    prob_matrix = time_stp * vibration_frequency * np.exp(-exponente)

    # Limitar probabilidades máximas a 1
    prob_matrix = np.minimum(prob_matrix, 1.0)

    return prob_matrix


def get_generation_probabilities_matrix(
    state: np.ndarray,
    paso_temporal: float,
    Electric_field: np.ndarray | float,
    temperatura: np.ndarray | float,
    factor_vecinos: float,
    factor_sin_vecinos: float,
    vibration_frequency: float,
    generation_energy: float,
    cte_red: float,
    gamma: float,
    neighbor_mode: str = "both",
) -> np.ndarray:
    """
    Calcula el mapa de probabilidades locales (0 a 1) para generar nuevas vacantes en la red,
    teniendo en cuenta la temperatura, el campo eléctrico y la topología de vecinos.
    """

    # 1. Máscara de posiciones libres
    free_mask = state == 0
    vecino_mask = np.zeros_like(state, dtype=bool)

    # 2. Lógica de Vecinos
    if neighbor_mode in ["horizontal", "both"]:
        left_neighbor = np.zeros_like(state, dtype=bool)
        left_neighbor[:, 1:] = state[:, :-1] == 1
        right_neighbor = np.zeros_like(state, dtype=bool)
        right_neighbor[:, :-1] = state[:, 1:] == 1
        vecino_mask |= left_neighbor | right_neighbor

    if neighbor_mode in ["vertical", "both"]:
        up_neighbor = np.zeros_like(state, dtype=bool)
        up_neighbor[1:, :] = state[:-1, :] == 1
        down_neighbor = np.zeros_like(state, dtype=bool)
        down_neighbor[:-1, :] = state[1:, :] == 1
        vecino_mask |= up_neighbor | down_neighbor

    free_with_vecino = free_mask & vecino_mask
    free_without_vecino = free_mask & (~vecino_mask)

    # 3. Probabilidad base
    prob_base_raw = calcular_probabilidad_generacion(
        time_stp=paso_temporal,
        electric_field=Electric_field,
        temp=temperatura,
        vibration_frequency=vibration_frequency,
        generation_energy=generation_energy,
        cte_red=cte_red,
        gamma=gamma,
    )

    # Si la calculadora devolvió un escalar (float), lo expandimos a una matriz del mismo tamaño que 'state'
    if not isinstance(prob_base_raw, np.ndarray) or prob_base_raw.ndim == 0:
        prob_base_matrix = np.full(state.shape, prob_base_raw)
    else:
        prob_base_matrix = prob_base_raw

    # 4. Aplicar factores condicionales (Topología)
    prob_final = np.zeros_like(prob_base_matrix)
    prob_final[free_with_vecino] = prob_base_matrix[free_with_vecino] * factor_vecinos
    prob_final[free_without_vecino] = prob_base_matrix[free_without_vecino] * factor_sin_vecinos

    # Saturamos a 1.0 por seguridad
    prob_final = np.minimum(prob_final, 1.0)

    return prob_final
