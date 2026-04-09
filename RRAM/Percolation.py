# Importamos las librerias necesarias

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement


# Compruebo primero si hay caminables en la primera y última columna de la matriz
def is_path(configuration_matrix: np.ndarray) -> bool:
    """
    Determines if there is a path from the leftmost column to the rightmost column
    in a given configuration matrix using the A* pathfinding algorithm.
    Args:
        configuration_matrix (np.ndarray): A 2D numpy array representing the configuration matrix.
    Returns:
        bool: True if there is a path from the leftmost column to the rightmost column, False otherwise.
    """

    # obtengo las dimensiones de la matriz
    _, Ejey = configuration_matrix.shape

    # Añado a los datos una primera columna llena de 1 y una última columna llena de 1
    # para considerar los nodos de inicio y fin
    configuration_matrix = np.insert(configuration_matrix, 0, 1, axis=1)
    configuration_matrix = np.insert(configuration_matrix, Ejey + 1, 1, axis=1)

    grid = Grid(matrix=configuration_matrix)

    # Recorro los nodos de inicio y fin para ver si hay camino
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

    node_start = grid.node(0, 0)
    node_end = grid.node(Ejey+1, 0)

    path, _ = finder.find_path(node_start, node_end, grid)

    if len(path) > 0:
        # Para dibujar el camino
        # print(grid.grid_str(path=path, start=node_start, end=node_end))
        return True
    else:
        return False


def Obtenin_Paths(configuration_matrix: np.ndarray) -> list:
    """
    Finds all possible paths from the first column to the last column in a given configuration matrix using the A* algorithm.
    Args:
        configuration_matrix (np.ndarray): A 2D numpy array representing the configuration matrix where '1' indicates a traversable cell and '0' indicates an obstacle.
    Returns:
        list: A list of numpy arrays, each containing tuples representing the coordinates of the path from the first column to the last column.
    """
    # obtengo las dimensiones de la matriz
    _, Ejey = configuration_matrix.shape

    # Lista para almacenar todos los caminos
    all_paths_list = []

    # Compruebo si hay trampas en la primera y última columna
    if 1 in configuration_matrix[:, 0] and 1 in configuration_matrix[:, -1]:
        # obtengo las posiciones de los 1 en la primera columna
        start = np.where(configuration_matrix[:, 0] == 1)[0]

        # Creo la lista de nodos de inicio sabiendo que todos los nodos de inicio están en la primera columna
        start = [(i, 0) for i in start]

        # obtengo las posiciones de los 1 en la última columna
        end = np.where(configuration_matrix[:, -1] == 1)[0]

        for k in [1, 2]:
            # obtengo las posiciones de los 1 en la última columna
            end = np.where(configuration_matrix[:, -k] == 1)[0]

            # Creo la lista de nodos de fin sabiendo que todos los nodos de fin están en la última columna en forma de grid node
            end = [(i, Ejey - k) for i in end]

            # Recorro los nodos de inicio y fin para ver si hay camino
            finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

            for i in start:
                for j in end:
                    grid = Grid(matrix=configuration_matrix)

                    node_start = grid.node(i[1], i[0])
                    node_end = grid.node(j[1], j[0])

                    path, runs = finder.find_path(node_start, node_end, grid)
                    if len(path) > 0:
                        # Convert path to a numpy array of tuples
                        path_tuples = np.array([(node.x, node.y) for node in path])
                        all_paths_list.append(path_tuples)

    return all_paths_list
