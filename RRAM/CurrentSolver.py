from numpy.typing import NDArray
import networkx as nx
import numpy as np
import math
import os

from scipy.constants import elementary_charge, Boltzmann, epsilon_0
from RRAM import Representate as rp
from RRAM import Constants as cte


def Clean_state_matrix(
    config_matrix: np.ndarray, min_size: int = 60, plot_resmatrix: bool = False
) -> tuple[np.ndarray, nx.Graph]:
    """
    Limpia la matriz de estado eliminando clústeres aislados y pseudo-filamentos.
    Garantiza que solo sobreviven las estructuras ancladas a AMBOS electrodos.
    """

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
                        actual_state_clean_grid.add_edge((i, j), (ni, nj))  # agregamos arista bidireccional

    # Elimino los nodos que no tienen aristas (puntos totalmente aislados)
    actual_state_clean_grid.remove_nodes_from(list(nx.isolates(actual_state_clean_grid)))

    # -------------------------------------------------------------------------
    # NUEVO FILTRO : Eliminar filamentos sin terminar y agrupaciones flotantes
    # -------------------------------------------------------------------------
    componentes = list(nx.connected_components(actual_state_clean_grid))
    validos = []

    for c in componentes:
        # Obtenemos todas las columnas (coordenada j) que ocupa este componente
        columnas = {n[1] for n in c}

        # CONDICIÓN FÍSICA: El componente DEBE tocar ambos electrodos (0 y W-1)
        # Opcionalmente, podemos mantener la condición de min_size para mayor robustez
        if (0 in columnas) and ((W - 1) in columnas):
            validos.append(c)

    # Reconstruimos el grafo solo con los componentes que percola
    nodos_finales = set().union(*validos)
    actual_state_clean_grid = actual_state_clean_grid.subgraph(nodos_finales).copy()

    # Nodos con grado 1 (hojas), excluyendo primera y última columna para "podar" ramas muertas
    while True:
        # Encontrar nodos hoja excluyendo primera y última columna (electrodos)
        leaf_nodes = [n for n, d in actual_state_clean_grid.degree() if d == 1 and n[1] not in (0, W - 1)]

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

    return actual_state_clean_matrix, actual_state_clean_grid


def Clasificar_CF(
    G: nx.Graph,
    x_max: int,
    y_max: int,
    filamentos_ranges: list[tuple[int, int]],
) -> dict:
    """
    Classifies starting nodes of a graph into filaments and determines if they percolate.
    This function takes a graph `G` and classifies its starting nodes (nodes with y-coordinate 0)
    into filaments based on their x-coordinate ranges. It also checks if there is a path from each
    starting node to any of the ending nodes (nodes with y-coordinate `x_max - 1`).
    Args:
        G (nx.Graph): The graph representing the system.
        x_max (int): The maximum x-coordinate value in the graph.
        y_max (int): The maximum y-coordinate value in the graph (not used in this function).
        filamentos_ranges (list[tuple[int, int]]): A list of tuples representing the x-coordinate
            ranges for each filament. Each tuple is of the form (fila_min, fila_max).
    Returns:
        dict: A dictionary where each key is a starting node, and the value is another dictionary
        with the following keys:
            - "filamento" (int): The filament number to which the node belongs.
            - "percola" (bool): True if there is a path from the node to any ending node, False otherwise.
    Raises:
        ValueError: If a starting node cannot be assigned to any filament based on the provided ranges.
    Example:
        >>> G = nx.grid_2d_graph(5, 5)
        >>> x_max = 5
        >>> y_max = 5
        >>> filamentos_ranges = [(0, 2), (3, 4)]
        >>> Clasificar_CF(G, x_max, y_max, filamentos_ranges)
        {
            (0, 0): {"filamento": 1, "percola": True},
            (1, 0): {"filamento": 1, "percola": True},
            ...
        }
    """

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
        existe = any(info["filamento"] == i and info["percola"] for info in resultados.values())
        Percola_list_bool.append(existe)

    return Percola_list_bool


def Eliminar_filamentos_incompletos(grid_limpio, filamentos_ranges, percola_bools, W=None):
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
    # Si no se pasa W, lo calcula automáticamente del tamaño real de la matriz

    # Si no se pasa W, calcularlo del grafo
    if W is None:
        nodos = list(grid_limpio.nodes())
        if nodos:
            max_dim = max(max(n[0], n[1]) for n in nodos)
            W = max_dim + 1
        else:
            raise ValueError("El grafo está vacío y no se proporcionó la dimensión W")

    G_limpio = grid_limpio.copy()

    for idx, percola in enumerate(percola_bools):
        if not percola:
            fila_min, fila_max = filamentos_ranges[idx]
            nodos_a_borrar = [(i, j) for i in range(fila_min, fila_max + 1) for j in range(W)]
            G_limpio.remove_nodes_from(nodos_a_borrar)

    # Crear matriz vacía de ceros para la matriz resistencia
    CF_matrix = np.zeros((W, W), dtype=int)

    # Poner 1 en las posiciones de los nodos que siguen en G
    for i, j in G_limpio.nodes():
        CF_matrix[i, j] = 1

    # rp.RepresentateState(CF_matrix, 0.00, os.getcwd() + "/Matriz_resistencia.pdf")

    return CF_matrix


def calcular_resistencia(CF_matrix, ohm_resistence):
    """
    Calcula la resistencia total de una matriz de formación de filamentos conductores (CF_matrix).
    Este método asume que cada columna de la matriz representa un conjunto de resistencias en paralelo.
    La resistencia total se calcula sumando las resistencias paralelas de cada columna.
    Args:
        CF_matrix (numpy.ndarray): Matriz donde cada elemento representa la presencia (1) o ausencia (0)
                                    de un filamento conductor.
        ohm_resistence (float, optional): Resistencia en ohmios asociada a cada filamento conductor.
                                           Por defecto es 11.5 ohmios.
    Returns:
        float: Resistencia total calculada a partir de la matriz CF_matrix.
    """
    total_resistance = 0.0
    Ny, Nx = CF_matrix.shape
    for j in range(1, Nx - 1):
        fil_indices = np.where(CF_matrix == 1)[0]
        N_total_columna = len(fil_indices)

        if N_total_columna == 0:
            continue

        # R equivalente de la columna (N resistencias en paralelo)
        R_col = ohm_resistence / N_total_columna

        # Se suma la resistencia paralela a la resistencia total
        total_resistance += R_col
    return total_resistance


def OmhCurrent(potential: float, resistance_matrix: NDArray, ohm_resistence: float) -> tuple[float, float]:
    """
    Calculates the Ohmic current based on the given parameters.

    Parameters:
    - potential (float): The voltage difference across the circuit.
    - resistance_matrix (np.array): The matrix representing the conductive filaments.
    - ohm_resistence (float): The resistance value of a single cell.

    Returns:
    - tuple[float, float]: A tuple containing the calculated Ohmic current and the total resistance.
    """

    # Se calcula la resistencia total usando el valor explícito que se le pasa
    total_resistance = calcular_resistencia(resistance_matrix, ohm_resistence)

    # Se calcula la corriente de Ohm (I = V/R)
    intensidad_ohmica = potential / total_resistance

    return intensidad_ohmica, total_resistance


# En RRAM/CurrentSolver.py
def Poole_Frenkel(
    temperature: float, E_field: float, pb_metal_insul: float, permitividad_relativa: float, I_0: float
) -> float:
    """
    Calcula la corriente mediante el mecanismo de conducción Poole-Frenkel.

    El efecto Poole-Frenkel describe la emisión de portadores de carga desde trampas
    en un material aislante bajo la influencia de un campo eléctrico alto. Este modelo
    es importante en la caracterización de materiales dieléctricos y semiconductores.

    Args:
        temperature (float): Temperatura absoluta en Kelvin [K]
        E_field (float): Campo eléctrico aplicado en V/m [V/m]
        pb_metal_insul (float): Barrera de potencial metal-aislante en eV [eV]
        permitividad_relativa (float): Permitividad relativa del material dieléctrico [adimensional]
        I_0 (float): Corriente de referencia o prefactor en Amperios [A]

    Returns:
        float: Corriente Poole-Frenkel calculada en Amperios [A]

    Fórmula:
        I_PF = I_0 * E * exp((β - φ_b) / (k_B * T))
        donde β = sqrt(e³ / (π * ε_0 * ε_r)) * sqrt(E)
    """

    k_b_ev = Boltzmann / elementary_charge
    beta = math.sqrt(abs(E_field)) * math.sqrt(elementary_charge / (epsilon_0 * permitividad_relativa * math.pi))

    exponencial = math.exp((beta - pb_metal_insul) / (k_b_ev * temperature))
    I_poole_frenkel = I_0 * E_field * exponencial

    return I_poole_frenkel
