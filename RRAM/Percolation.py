# Importamos las librerias necesarias

import numpy as np
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement


# Compruebo primero si hay caminables en la primera y última columna de la matriz
def is_path(configuration_matrix: np.ndarray) -> bool:

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


def num_path(configuration_matrix: np.ndarray) -> bool:

    # obtengo las dimensiones de la matriz
    _, Ejey = configuration_matrix.shape

    # Creo una variable para saber si hay camino
    ExistPath = False

    # Añado a los datos una primera columna llena de 1 y una última columna llena de 1
    configuration_matrix = np.insert(configuration_matrix, 0, 1, axis=1)
    configuration_matrix = np.insert(configuration_matrix, Ejey + 1, 1, axis=1)
    print(configuration_matrix)

    # Creo el grid a partir de la matriz de configuración

    # Compruebo si hay trampas en la primera y última columna
    if 1 in configuration_matrix[:, 0] and 1 in configuration_matrix[:, -1]:
        # obtengo las posiciones de los 1 en la primera columna
        start = np.where(configuration_matrix[:, 0] == 1)[0]

        # Creo la lista de nodos de inicio sabiendo que todos los nodos de inicio están en la primera columna
        start = [(i, 0) for i in start]

        # obtengo las posiciones de los 1 en la última columna
        end = np.where(configuration_matrix[:, -1] == 1)[0]

        # Creo la lista de nodos de fin sabiendo que todos los nodos de fin están en la última columna en forma de grid node
        end = [(i, Ejey - 1) for i in end]

        # Recorro los nodos de inicio y fin para ver si hay camino
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)

        for i in start:
            for j in end:
                grid = Grid(matrix=configuration_matrix)

                node_start = grid.node(i[1], i[0])
                node_end = grid.node(j[1], j[0])

                path, runs = finder.find_path(node_start, node_end, grid)
                if len(path) > 0:
                    print(grid.grid_str(path=path, start=node_start, end=node_end))
                    ExistPath = True
                    break
            if len(path) > 0:
                break
        return ExistPath
    else:
        return ExistPath


if __name__ == "__main__":

    import Representate as rp

    Ejex = 50
    Ejey = 50

    # matriz = [[0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
    #           [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    #           [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
    #           [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    #           [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    #           [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    #           [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    #           [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    #           [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
    #           [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    #           [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    #           [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    #           [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    #           [0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    #           [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    #           [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],
    #           [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    #           [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
    #           [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    #           [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0]]

    # matriz = np.array(matriz)

    matriz = np.zeros((Ejex, Ejey))

    # relleno la matriz con 1 en posiciones aleatorias
    for i in range(Ejex):
        for j in range(Ejey):
            matriz[i][j] = np.random.choice([0, 1])

    rp.RepresentateState(matriz, "prueba.png")

    # Imprimo un salto de linea
    # print(matriz, '\n')
    if is_path(matriz):
        print('Hay camino')
    else:
        print('No hay camino')
