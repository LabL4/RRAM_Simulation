import numpy as np  # type: ignore
from cycler import V
# import math

from RRAM import Constants as cte


def get_required_param(kwargs, param_name):
    if param_name not in kwargs:
        raise ValueError(f"El parámetro '{param_name}' es obligatorio")
    return kwargs[param_name]


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
    posiciones_unos = np.random.choice(
        Eje_x * Eje_y, num_trampas, replace=False, p=pesos_flat / np.sum(pesos_flat)
    )

    # Assign the value 1 to the selected positions
    for pos in posiciones_unos:
        fila, columna = divmod(pos, Eje_y)
        InitialState[fila, columna] = 1

    return InitialState


def Generate(time_stp: float, electric_field: float, temp: float, **kwargs) -> float:
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

    # Obtengo las constantes necesarias para el cálculo
    if kwargs:
        t_0 = float(get_required_param(kwargs, "vibration_frequency"))
        E_a = float(get_required_param(kwargs, "activation_energy"))
        cte_red = float(get_required_param(kwargs, "cte_red"))
        gamma = float(get_required_param(kwargs, "gamma"))
    else:
        # Aquí reemplaza con tus constantes globales si las tienes
        t_0 = cte.t_0
        E_a = cte.E_a
        cte_red = cte.cte_red
        gamma = cte.gamma

    exponente = (E_a - (gamma * cte_red * electric_field)) / (cte.k_b_ev * temp)
    prob_generacion = time_stp * t_0 * (np.exp(-exponente))

    return prob_generacion


def Generate_vectorized(
    time_stp: float, electric_field_matrix: np.ndarray, temp: float, **kwargs
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

    if kwargs:
        t_0 = float(get_required_param(kwargs, "vibration_frequency"))
        E_a = float(get_required_param(kwargs, "activation_energy"))
        cte_red = float(get_required_param(kwargs, "cte_red"))
        gamma = float(get_required_param(kwargs, "gamma"))
    else:
        # Aquí reemplaza con tus constantes globales si las tienes
        t_0 = cte.t_0
        E_a = cte.E_a
        cte_red = cte.cte_red
        gamma = cte.gamma

    exponente = (E_a - (gamma * cte_red * electric_field_matrix)) / (cte.k_b_ev * temp)
    prob_matrix = time_stp * t_0 * np.exp(-exponente)

    # Limitar probabilidades máximas a 1
    prob_matrix = np.minimum(prob_matrix, 1.0)

    return prob_matrix


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


def generate_oxygen(oxygen_state: np.ndarray, num_oxygen: int):
    eje_y = oxygen_state.shape[1]

    # Generar todas las coordenadas y al mismo tiempo
    y_indices = np.random.randint(0, eje_y, size=num_oxygen)

    # Generar todas las probabilidades de una sola vez
    rand_vals = np.random.rand(num_oxygen)

    # Crear una máscara para las posiciones donde se colocarán oxígenos
    mask = (oxygen_state[y_indices, 0] == 0) & (rand_vals < 0.35)

    # Asignar oxígeno en las posiciones seleccionadas
    oxygen_state[y_indices[mask], 0] = 1

    return oxygen_state


def generate_oxigen_old(oxygen_state: np.ndarray, num_oxygen: int):
    """
    Generates random oxygen positions in the given oxygen state matrix.

    Args:
        oxigen_state (np.ndarray): The current oxygen state matrix.
        num_oxigen (int): The number of oxygen to generate.

    Returns:
        np.ndarray: The updated oxygen state matrix with the generated oxygen positions.
    """

    eje_y = oxygen_state.shape[1]
    y = np.zeros(num_oxygen, dtype=int)

    for i in range(num_oxygen):
        # Genero las coordenadas aleatorias para el eje y donde habrá un oxígeno
        y[i] = np.random.randint(0, eje_y)

    if num_oxygen == 1:
        prob = 0.5  # 0.75
    elif num_oxygen == 1:
        prob = 0.7  # 0.9
    elif num_oxygen == 3:
        prob = 1

    # Itero sobre cada par coordenada para asignar el valor de 1 que representa que se generó un oxígeno en esa posición
    try:
        for i in range(num_oxygen):
            random_number = np.random.rand()
            if oxygen_state[y[i], 0] == 0 and random_number < prob:
                oxygen_state[y[i], 0] = 1
    except ValueError:
        print(
            "Error en la generación de oxígenos, la probabilidad es",
            prob,
            "que corresponde a",
            num_oxygen,
            "oxígenos.",
        )

    # Devuelvo la matriz con los oxígenos generados
    return oxygen_state
