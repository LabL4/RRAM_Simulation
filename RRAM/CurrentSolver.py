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
        # Obtenemos todas las filas X (coordenada i) que ocupa este componente
        # CONDICIÓN FÍSICA: el filamento debe tocar ambos electrodos en eje X (filas 0 y H-1)
        filas = {n[0] for n in c}
        if (0 in filas) and ((H - 1) in filas):
            validos.append(c)

    # Reconstruimos el grafo solo con los componentes que percolan
    nodos_finales = set().union(*validos)
    actual_state_clean_grid = actual_state_clean_grid.subgraph(nodos_finales).copy()

    # Nodos con grado 1 (hojas), excluyendo electrodos (filas 0 y H-1) para podar ramas muertas
    while True:
        leaf_nodes = [n for n, d in actual_state_clean_grid.degree() if d == 1 and n[0] not in (0, H - 1)]

        if not leaf_nodes:
            break

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
    # Nodos de inicio: fila X=0 (primer electrodo). Nodos de fin: fila X=x_max-1 (segundo electrodo).
    nodos_inicio = [n for n in G.nodes() if n[0] == 0]
    nodos_fin = [n for n in G.nodes() if n[0] == x_max - 1]

    # Clasificación de nodos de inicio según su posición Y (columna) en los rangos de filamento
    nodos_inicio_clasificados = {}
    for nodo in nodos_inicio:
        columna_y = nodo[1]  # posición Y (transversal) del nodo
        filamento_asignado = None
        for idx, (col_min, col_max) in enumerate(filamentos_ranges, start=0):
            if col_min <= columna_y <= col_max:
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


def Eliminar_filamentos_incompletos(grid_limpio, filamentos_ranges, percola_bools, H=None, W=None):
    """
    Elimina los nodos y aristas de los rangos donde el filamento no se ha formado (no percola).

    Args:
        grid_limpio : nx.Graph
            Grafo original copiado, tamaño completo.
        filamentos_ranges : list of tuples
            Lista de rangos de fila [(fila_min, fila_max), ...] para cada filamento.
        percola_bools : list of bool
            Lista booleana que indica si cada filamento percola.
        H : int, opcional
            Número de filas de la matriz (eje Y, ancho transversal). Se calcula del grafo si no se pasa.
        W : int, opcional
            Número de columnas de la matriz (eje X, distancia entre electrodos). Se calcula del grafo si no se pasa.

    Returns:
        CF_matrix : np.ndarray
            Matriz (H x W) con los nodos (y sus aristas) solo de los filamentos completos.
    """
    # Calcular H y W por separado desde las coordenadas de los nodos (fila=Y, columna=X)
    if H is None or W is None:
        nodos = list(grid_limpio.nodes())
        if nodos:
            H = max(n[0] for n in nodos) + 1  # max fila + 1 = y_size
            W = max(n[1] for n in nodos) + 1  # max columna + 1 = x_size
        else:
            raise ValueError("El grafo está vacío y no se proporcionaron las dimensiones H y W")

    G_limpio = grid_limpio.copy()

    for idx, percola in enumerate(percola_bools):
        if not percola:
            fila_min, fila_max = filamentos_ranges[idx]
            nodos_a_borrar = [(i, j) for i in range(fila_min, fila_max + 1) for j in range(W)]
            G_limpio.remove_nodes_from(nodos_a_borrar)

    # Crear matriz rectangular (H filas = eje Y, W columnas = eje X)
    CF_matrix = np.zeros((H, W), dtype=int)

    # Poner 1 en las posiciones de los nodos que siguen en G
    for i, j in G_limpio.nodes():
        CF_matrix[i, j] = 1

    return CF_matrix


def calcular_resistencia(CF_matrix, ohm_resistence):
    """
    Calcula la resistencia total de la CF_matrix modelando la corriente a lo largo del eje X.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección de la corriente, entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal.

    Cada fila X representa una capa de resistencias en paralelo (las celdas Y con filamento).
    La resistencia total es la suma serie de todas las capas X.

    Args:
        CF_matrix (np.ndarray): Matriz (x_size, y_size), 1=filamento, 0=vacío.
        ohm_resistence (float): Resistencia de una celda individual [Ohm].
    Returns:
        float: Resistencia total [Ohm].
    """
    total_resistance = 0.0
    x_size, y_size = CF_matrix.shape

    for i in range(x_size):
        row_data = CF_matrix[i, :]
        N_paralelo = int(np.sum(row_data))  # celdas Y con filamento en esta fila X

        if N_paralelo == 0:
            continue

        # R equivalente de la fila (N resistencias en paralelo)
        R_fila = ohm_resistence / N_paralelo
        total_resistance += R_fila

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


def limitar_grosor_filamentos(
    matriz_state: np.ndarray,
    matriz_solo_filamentos: np.ndarray,
    centros_calculados: list,
    casillas_mantener: int,
    rangos_CF: list,
    valor_oxido: int = 0,  # <-- Ahora el valor por defecto es 0 (el óxido en tu matriz)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorta el grosor de los filamentos eliminando las vacantes que se alejan
    demasiado de su centro, respetando los límites físicos (rangos) de cada filamento.

    Argumentos:
    - matriz_state: Matriz cuadrada de la simulación (0=óxido, 1=vacantes).
    - matriz_solo_filamentos: Matriz que SOLO contiene las vacantes que forman filamento.
    - centros_calculados: Lista con la fila central de cada filamento.
    - casillas_mantener: Cuántas filas por encima y por debajo del centro se conservan.
    - rangos_CF: Lista de tuplas indicando el límite (ymin, ymax) de cada filamento.
    - valor_oxido: Valor numérico para rellenar el hueco dejado (0 por defecto).

    Retorna:
    - (matriz_state_recortada, matriz_filamentos_recortada)
    """
    # 1. Copias de seguridad para no alterar las matrices originales
    m_state_recortada = np.copy(matriz_state)
    m_filamentos_recortada = np.copy(matriz_solo_filamentos)

    # 2. Recorremos cada filamento con su centro Y y su rango Y
    x_size, y_size = m_filamentos_recortada.shape

    for centro, rango in zip(centros_calculados, rangos_CF):
        # Si no hay filamento formado en este rango, pasamos al siguiente
        if centro is None or rango is None:
            continue

        # centro y rango son posiciones en el eje Y (columnas, dirección transversal)
        col_min, col_max = rango

        # 3. Fronteras Y permitidas para el grosor del filamento
        limite_inferior = centro - casillas_mantener
        limite_superior = centro + casillas_mantener

        # 4. Escaneamos columnas Y del rango de este filamento
        for col_y in range(col_min, col_max + 1):
            if col_y < 0 or col_y >= y_size:
                continue  # evitar IndexError en matrices rectangulares

            # Si la columna Y está FUERA del grosor permitido alrededor del centro
            if col_y < limite_inferior or col_y > limite_superior:
                # Filas X con filamento en esta columna Y
                filas_vacantes = np.where(m_filamentos_recortada[:, col_y] == 1)[0]

                if len(filas_vacantes) > 0:
                    m_filamentos_recortada[filas_vacantes, col_y] = valor_oxido
                    m_state_recortada[filas_vacantes, col_y] = valor_oxido

    return m_state_recortada, m_filamentos_recortada
