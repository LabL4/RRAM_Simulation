from numpy.typing import NDArray
import networkx as nx
import numpy as np
import math
import os

from scipy.constants import elementary_charge, Boltzmann, epsilon_0
from RRAM import Representate as rp
from RRAM import Constants as cte


def Clean_state_matrix(
    config_matrix: np.ndarray, min_size: int = 20, plot_resmatrix: bool = False
) -> tuple[np.ndarray, nx.Graph]:
    # Preparar el grafo no dirigido (A-B es igual que B-A)
    actual_state_clean_grid = nx.Graph()
    grid = np.array(config_matrix)
    H, W = grid.shape
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # movimientos ortogonales

    # Creo el grid de nodos a partir de la matriz de configuracion
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 1:  # solo celdas caminables
                actual_state_clean_grid.add_node((i, j))  # agregamos el nodo
                for di, dj in dirs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 1:
                        actual_state_clean_grid.add_edge(
                            (i, j), (ni, nj)
                        )  # agregamos arista bidireccional

    # Elimino los nodos que no tienen aristas
    actual_state_clean_grid.remove_nodes_from(
        list(nx.isolates(actual_state_clean_grid))
    )

    # ELimino componentes pequeños sueltos
    componentes = list(nx.connected_components(actual_state_clean_grid))
    validos = [c for c in componentes if len(c) > min_size]
    nodos_finales = set().union(*validos)
    actual_state_clean_grid = actual_state_clean_grid.subgraph(nodos_finales).copy()

    # Nodos con grado 1 (hojas), excluyendo primera y última columna hasta que no quede ningún nodo sin conexión
    while True:
        # Encontrar nodos hoja excluyendo primera y última columna
        leaf_nodes = [
            n
            for n, d in actual_state_clean_grid.degree()
            if d == 1 and n[1] not in (0, W - 1)
        ]

        # Si no hay más nodos hoja, salir del bucle
        if not leaf_nodes:
            break

        # Eliminar los nodos hoja encontrados
        actual_state_clean_grid.remove_nodes_from(leaf_nodes)

    # Crear matriz vacía de ceros para la matriz resistencia
    actual_state_clean_matrix = np.zeros((H, W), dtype=int)

    # Poner 1 en las posiciones de los nodos que siguen en G
    for i, j in actual_state_clean_grid.nodes():
        actual_state_clean_matrix[i, j] = 1

    if plot_resmatrix:
        rp.RepresentateState(
            actual_state_clean_matrix, 0.00, os.getcwd() + "/Matriz_resistencia.pdf"
        )

    return actual_state_clean_matrix, actual_state_clean_grid


def Clasificar_CF(
    G: nx.Graph,
    x_max: int,
    y_max: int,
    filamentos_ranges: list[tuple[int, int]],
) -> dict:
    # Nodos de origen y destino
    nodos_inicio = [n for n in G.nodes() if n[1] == 0]
    nodos_fin = [n for n in G.nodes() if n[1] == x_max - 1]

    # Clasificación de nodos de inicio en filamentos
    nodos_inicio_clasificados = {}
    for nodo in nodos_inicio:
        fila = nodo[0]
        filamento_asignado = None
        for idx, (fila_min, fila_max) in enumerate(filamentos_ranges, start=0):
            if fila_min <= fila <= fila_max:  # tolerancia ±0 por defecto
                filamento_asignado = idx + 1  # Filamentos numerados desde 1
                break
        if filamento_asignado is None:
            raise ValueError(f"Nodo {nodo} no pudo asignarse a ningún filamento")
        nodos_inicio_clasificados[nodo] = filamento_asignado

    resultados = {}
    for nodo, filamento in nodos_inicio_clasificados.items():
        percola = any(nx.has_path(G, nodo, t) for t in nodos_fin)
        resultados[nodo] = {"filamento": filamento, "percola": percola}

    return resultados


def Existe_filamentos(resultados, num_filamentos) -> list[bool]:
    """
    Comprueba si cada filamento tiene al menos un nodo de inicio que percola.

    Args:
        resultados : dict
            Diccionario { nodo: {"filamento": i, "percola": True/False} }
        num_filamentos : int
            Número total de filamentos

    Returns:
        lista_bool : list
            Lista de booleanos [F1, F2, ...] indicando si cada filamento tiene
            al menos un nodo que percola
    """
    Percola_list_bool = []

    for i in range(1, num_filamentos + 1):
        existe = any(
            info["filamento"] == i and info["percola"] for info in resultados.values()
        )
        Percola_list_bool.append(existe)

    return Percola_list_bool


def Eliminar_filamentos_incompletos(
    grid_limpio, filamentos_ranges, percola_bools, W: int = 40
):
    """
    Elimina los nodos y aristas de los rangos donde el filamento no se ha formado (no percola).

    Args:
        G : nx.Graph
            Grafo original copiado, tamaño completo.
        filamentos_ranges : list of tuples
            Lista de rangos de fila [(fila_min, fila_max), ...] para cada filamento.
        percola_bools : list of bool
            Lista booleana que indica si cada filamento percola.
        W : int representa el ancho del espacio de configuración (número de columnas).

    Returns:
        G_limpio : nx.Graph
            Grafo con los nodos (y sus aristas) solo de los filamentos completos.
    """
    G_limpio = grid_limpio.copy()

    for idx, percola in enumerate(percola_bools):
        if not percola:
            fila_min, fila_max = filamentos_ranges[idx]
            nodos_a_borrar = [
                (i, j) for i in range(fila_min, fila_max + 1) for j in range(W)
            ]
            G_limpio.remove_nodes_from(nodos_a_borrar)

    # Crear matriz vacía de ceros para la matriz resistencia
    CF_matrix = np.zeros((W, W), dtype=int)

    # Poner 1 en las posiciones de los nodos que siguen en G
    for i, j in G_limpio.nodes():
        CF_matrix[i, j] = 1

    # rp.RepresentateState(CF_matrix, 0.00, os.getcwd() + "/Matriz_resistencia.pdf")

    return CF_matrix


def calcular_resistencia(CF_matrix, ohm_resistence=11.5):
    total_resistance = 0.0
    # Sobre cada columna de la matriz
    for row in CF_matrix.T:
        # Se calcula la resistencia paralela de los elementos de la columna
        num_resistence = sum(row)
        parallel_resistance = 1 / (num_resistence / ohm_resistence)

        # Se suma la resistencia paralela a la resistencia total
        total_resistance += parallel_resistance

        # Se calcula la corriente Ohmica
    return total_resistance


def OmhCurrent(
    potential: float, resistance_matrix: NDArray, **kwargs
) -> tuple[float, float]:
    """
    Calculates the Ohmic current based on the given parameters.

    Parameters:
    - DDP (float): The voltage difference across the circuit.
    - config_state (np.array): The resistance matrix of the system.
    - ohm_resistence (float): The default resistance value.

    Returns:
    - float: The calculated Ohmic current.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función
        ohm_resistence = float(kwargs.get("ohm_resistence"))  # type: ignore
    else:
        ohm_resistence = 11.5  # Valor por defecto

    total_resistance = calcular_resistencia(resistance_matrix, ohm_resistence)

    intensidad_ohmica = potential / total_resistance

    # Se calcula la corriente Ohmica
    return intensidad_ohmica, total_resistance  # type: ignore


def Poole_Frenkel(temperature: float, E_field: float, **kwargs) -> float:
    """
    Calculate the current using the Poole-Frenkel effect.
    The Poole-Frenkel effect describes the lowering of the potential barrier in an insulator due to an applied electric field, which increases the current.
    Parameters:
    temperature (float): The temperature in Kelvin.
    E_field (float): The electric field in V/m.
    **kwargs: Optional keyword arguments for constants:
        - pb_metal_insul (float): Potential barrier between metal and insulator.
        - permitividad_relativa (float): Relative permittivity of the material.
        - I_0 (float): Pre-exponential factor.
    Returns:
    float: The current calculated using the Poole-Frenkel effect.
    """

    # Obtengo los valores de las constantes si las estoy pasando como argumentos
    if kwargs:
        # Obtengo el valor de las constantes que necesita la función hay q mejorarlo por si algun elemento fuera None
        potential_barrier = float(kwargs.get("pb_metal_insul"))  # type: ignore
        epsilon_r = float(kwargs.get("permitividad_relativa"))  # type: ignore
        I_0 = float(kwargs.get("I_0"))  # type: ignore
    else:
        potential_barrier = cte.pb_metal_insul
        epsilon_r = cte.permitividad_relativa
        I_0 = cte.I_0

    k_b_ev = Boltzmann / elementary_charge
    beta = math.sqrt(elementary_charge / (epsilon_0 * epsilon_r * math.pi))

    exponencial = math.exp(
        elementary_charge
        * (beta * math.sqrt(E_field) - potential_barrier)
        / (k_b_ev * temperature)
    )

    I_poole_frenkel = I_0 * E_field * exponencial

    return I_poole_frenkel
