import numpy as np

from RRAM import Constants as cte

k_b_ev = 8.617333262145e-5  # Boltzmann constant in eV/K


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
    custom_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calcula el mapa de probabilidades locales (0 a 1) para generar nuevas vacantes en la red.

    Si se proporciona custom_mask, sobreescribe neighbor_mode: la probabilidad base se
    multiplica por 1 en celdas con True y por 0 en celdas con False.
    Sin custom_mask se aplica la lógica de vecinos según neighbor_mode ("horizontal",
    "vertical" o "both"), usando factor_vecinos y factor_sin_vecinos.
    """

    free_mask = state == 0

    # Probabilidad base
    prob_base_raw = calcular_probabilidad_generacion(
        time_stp=paso_temporal,
        electric_field=Electric_field,
        temp=temperatura,
        vibration_frequency=vibration_frequency,
        generation_energy=generation_energy,
        cte_red=cte_red,
        gamma=gamma,
    )

    if not isinstance(prob_base_raw, np.ndarray) or prob_base_raw.ndim == 0:
        prob_base_matrix = np.full(state.shape, prob_base_raw)
    else:
        prob_base_matrix = prob_base_raw

    if custom_mask is not None:
        # True → factor 1 (probabilidad normal), False → factor 0 (probabilidad nula)
        prob_final = prob_base_matrix * free_mask * custom_mask.astype(float)
    else:
        # Lógica de vecinos estándar
        vecino_mask = np.zeros_like(state, dtype=bool)
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

        prob_final = np.zeros_like(prob_base_matrix)
        prob_final[free_mask & vecino_mask] = prob_base_matrix[free_mask & vecino_mask] * factor_vecinos
        prob_final[free_mask & ~vecino_mask] = prob_base_matrix[free_mask & ~vecino_mask] * factor_sin_vecinos

    return np.minimum(prob_final, 1.0)


def create_custom_mask(state, centros_CF, grosor_CF):
    """
    Crea una máscara para filamentos que ocupan TODA la fila horizontal del dispositivo.
    Cada filamento se define por su fila central y un grosor en filas adicionales
    arriba y abajo de dicha fila central.

    Args:
        state (np.ndarray): La matriz de estado actual (para obtener dimensiones).
        centros_CF (list of int): Lista de filas centrales de cada filamento.
        grosor_CF (int | list of int): Número de filas extra por encima y por debajo
            de la fila central. Puede ser un único entero (aplicado a todos los
            filamentos) o una lista con un valor por filamento.

    Returns:
        np.ndarray: Máscara booleana donde True indica que la celda pertenece a la
            banda horizontal de algún filamento (todas las columnas incluidas).
    """
    # Normalizar grosor_CF a lista, una entrada por filamento
    if isinstance(grosor_CF, (int, np.integer)):
        grosores = [int(grosor_CF)] * len(centros_CF)
    else:
        grosores = list(grosor_CF)
        if len(grosores) != len(centros_CF):
            raise ValueError("Debe haber un grosor asignado para cada centro de filamento.")

    filas, columnas = state.shape
    mask = np.zeros((filas, columnas), dtype=bool)

    for fila_central, grosor in zip(centros_CF, grosores):
        fila_inicio = max(0, fila_central - grosor)
        fila_fin = min(filas, fila_central + grosor + 1)
        # Todas las columnas de la banda de filas quedan activas
        mask[fila_inicio:fila_fin, :] = True

    return mask
