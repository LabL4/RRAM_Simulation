from numpy.typing import NDArray
import networkx as nx
import numpy as np
import math
import os

from scipy.constants import elementary_charge, Boltzmann, epsilon_0
from RRAM import Representate as rp
from RRAM import Constants as cte


def Generate_CF_matrix(
    config_matrix: np.ndarray, min_size: int = 20, plot_resmatrix: bool = False
) -> tuple[np.ndarray, nx.Graph]:
    # Preparar el grafo no dirigido (A-B es igual que B-A)
    G = nx.Graph()
    grid = np.array(config_matrix)
    H, W = grid.shape
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # movimientos ortogonales

    # Creo el grid de nodos a partir de la matriz de configuracion
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 1:  # solo celdas caminables
                G.add_node((i, j))  # agregamos el nodo
                for di, dj in dirs:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 1:
                        G.add_edge((i, j), (ni, nj))  # agregamos arista bidireccional

    # Elimino los nodos que no tienen aristas
    G.remove_nodes_from(list(nx.isolates(G)))

    # ELimino componentes pequeños sueltos
    componentes = list(nx.connected_components(G))
    validos = [c for c in componentes if len(c) > min_size]
    nodos_finales = set().union(*validos)
    G = G.subgraph(nodos_finales).copy()

    # Nodos con grado 1 (hojas), excluyendo primera y última columna hasta que no quede ningún nodo sin conexión
    while True:
        # Encontrar nodos hoja excluyendo primera y última columna
        leaf_nodes = [n for n, d in G.degree() if d == 1 and n[1] not in (0, W - 1)]

        # Si no hay más nodos hoja, salir del bucle
        if not leaf_nodes:
            break

        # Eliminar los nodos hoja encontrados
        G.remove_nodes_from(leaf_nodes)

    # Crear matriz vacía de ceros para la matriz resistencia
    CF_matrix = np.zeros((H, W), dtype=int)

    # Poner 1 en las posiciones de los nodos que siguen en G
    for i, j in G.nodes():
        CF_matrix[i, j] = 1

    if plot_resmatrix:
        rp.RepresentateState(CF_matrix, 0.00, os.getcwd() + "/Matriz_resistencia.pdf")

    return CF_matrix, G


def Clasificar_CF(
    G: nx.Graph,
    x_max: int,
    y_max: int,
    filamentos_ranges: list[tuple[int, int]],
    toleracia: int = 3,
) -> dict:
    # Nodos de origen y destino
    nodos_inicio = [n for n in G.nodes() if n[1] == 0]
    nodos_fin = [n for n in G.nodes() if n[1] == x_max - 1]

    # Clasificación de nodos de inicio en filamentos
    nodos_inicio_clasificados = {}
    for nodo in nodos_inicio:
        fila = nodo[0]
        filamento_asignado = None
        for idx, (fila_min, fila_max) in enumerate(filamentos_ranges, start=1):
            # TODO: debatible, poner directamente todo el rango? que son 4 bn pero con dos ya daría problemas
            if (
                fila_min - toleracia <= fila <= fila_max + toleracia
            ):  # tolerancia ±3 por defecto
                filamento_asignado = idx
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
    lista_bool = []

    for i in range(1, num_filamentos + 1):
        existe = any(
            info["filamento"] == i and info["percola"] for info in resultados.values()
        )
        lista_bool.append(existe)

    return lista_bool


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
        ohm_resistence = 11.5

    # Initialize total resistance
    total_resistance = 0
    parallel_resistance = 0

    # Sobre cada columna de la matriz
    for row in resistance_matrix.T:
        # Se calcula la resistencia paralela de los elementos de la columna
        num_resistence = sum(row)
        parallel_resistance = 1 / (num_resistence / ohm_resistence)

        # for i in range(num_resistence):
        #     parallel_resistance += 1 / ohm_resistence

        # Se suma la resistencia paralela a la resistencia total
        total_resistance += parallel_resistance

        # Se calcula la corriente Ohmica
    return potential / total_resistance, total_resistance  # type: ignore


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
