# Importamos las librerias necesarias

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement


# Comprueba si existe percolación desde el electrodo X=0 hasta el electrodo X=x_size-1
def is_path(configuration_matrix: np.ndarray) -> bool:
    """
    Determina si existe un camino conductor desde el primer electrodo (fila X=0)
    hasta el segundo electrodo (fila X=x_size-1) usando el algoritmo A*.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal.

    Se insertan filas centinela de 1s en los extremos del eje X para que A* pueda
    conectar desde cualquier celda Y en X=0 hasta cualquier celda Y en X=x_size-1.

    Args:
        configuration_matrix (np.ndarray): Matriz 2D (x_size, y_size). 1=caminable, 0=bloqueado.
    Returns:
        bool: True si existe percolación entre los dos electrodos, False en caso contrario.
    """
    x_size_mat, _ = configuration_matrix.shape  # x_size = shape[0] (filas = eje X)

    # Insertar filas centinela de 1s en los extremos del eje X (electrodos virtuales)
    configuration_matrix = np.insert(configuration_matrix, 0, 1, axis=0)
    configuration_matrix = np.insert(configuration_matrix, x_size_mat + 1, 1, axis=0)

    grid = Grid(matrix=configuration_matrix)
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

    # En pathfinding: node(x=col, y=row). Buscamos de fila 0 a fila x_size_mat+1.
    node_start = grid.node(0, 0)
    node_end = grid.node(0, x_size_mat + 1)

    path, _ = finder.find_path(node_start, node_end, grid)
    return len(path) > 0


def Obtenin_Paths(configuration_matrix: np.ndarray) -> list:
    """
    Encuentra todos los caminos posibles desde la primera fila X (electrodo 0)
    hasta la última fila X (electrodo x_size-1) usando el algoritmo A*.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos.
        shape[1] = eje Y (columnas) = ancho transversal.

    Args:
        configuration_matrix (np.ndarray): Matriz 2D (x_size, y_size). 1=caminable, 0=bloqueado.
    Returns:
        list: Lista de arrays con las coordenadas (col, row) de cada camino encontrado.
    """
    x_size_mat, _ = configuration_matrix.shape  # shape[0] = x_size (filas = eje X)

    all_paths_list = []

    # Solo buscamos caminos si hay filamento en la primera y última fila X (electrodos)
    if 1 not in configuration_matrix[0, :] or 1 not in configuration_matrix[-1, :]:
        return all_paths_list

    # Nodos de inicio: columnas Y en la fila X=0 donde hay filamento
    start_cols = np.where(configuration_matrix[0, :] == 1)[0]
    start_nodes = [(0, j) for j in start_cols]  # (row=0, col=j)

    for k in [1, 2]:
        # Nodos de fin: columnas Y en las últimas filas X donde hay filamento
        end_cols = np.where(configuration_matrix[-k, :] == 1)[0]
        end_nodes = [(x_size_mat - k, j) for j in end_cols]  # (row=x_size-k, col=j)

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

        for s in start_nodes:
            for e in end_nodes:
                grid = Grid(matrix=configuration_matrix)
                # pathfinding: node(x=col, y=row)
                node_start = grid.node(s[1], s[0])
                node_end = grid.node(e[1], e[0])

                path, _ = finder.find_path(node_start, node_end, grid)
                if len(path) > 0:
                    path_tuples = np.array([(node.x, node.y) for node in path])
                    all_paths_list.append(path_tuples)

    return all_paths_list
