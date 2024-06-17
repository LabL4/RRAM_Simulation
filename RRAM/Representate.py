# Description: This file contains the code to generate the mesh of the RRAM.
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.colors import LinearSegmentedColormap


def RepresentateState(matriz: np.ndarray, filename: str = "grafica.png") -> None:
    """
    Represent the state of a matrix as a colored plot.

    Parameters:
    - matriz (np.ndarray): The input matrix to be represented.
    - filename (str, optional): The name of the file to save the plot. Default is "grafica.png".

    Returns:
    None
    """
    nrows, ncols = matriz.shape

    x = np.arange(ncols)
    y = np.arange(nrows)

    fig, ax = plt.subplots()

    # Crear un mapa de colores personalizado
    colors = [
        (1, 1, 1),                  # Color para el valor 0 que representa que No hay trampa
        (0.478, 0.627, 0.870),      # Color para el valor 1 que representa que hay trampa (azul)
    ]
    if np.all(matriz == 1):
        colors = list(reversed(colors))

    cmap_name = "my_list"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    # Graficar la matriz
    ax.pcolormesh(
        x,
        y,
        matriz,
        shading="nearest",
        vmin=matriz.min(),
        vmax=matriz.max(),
        cmap=cmap,
    )

    # Configurar las marcas de los ejes
    major_ticks = np.arange(0, nrows, 1)
    minor_ticks = np.arange(0, nrows, 0.5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # Establecer relación de aspecto cuadrada
    ax.set_aspect("equal")

    # Invertir el eje y para que el punto (0, 0) esté en la esquina superior izquierda
    ax.invert_yaxis()

    # Coloco las etiquetas del eje x en la parte superior
    ax.xaxis.tick_top()

    # Guardar la imagen
    # plt.savefig(filename)

    return None


def RepresentateStateOpt(matriz: np.ndarray, filename: str = "grafica.png") -> None:
    """
    Represent the state of a matrix as a colored plot.

    Parameters:
    - matriz (np.ndarray): The input matrix to be represented.
    - filename (str, optional): The name of the file to save the plot. Default is "grafica.png".

    Returns:
    None
    """
    # Crear una figura y un eje con plt.subplots()
    fig, ax = plt.subplots()

    # Crear un mapa de colores personalizado
    colors = [
        (1, 1, 1),                  # Color para el valor 0 que representa que No hay trampa
        (0.478, 0.627, 0.870),      # Color para el valor 1 que representa que hay trampa (azul)
    ]
    if np.all(matriz == 1):
        colors = list(reversed(colors))

    cmap_name = "my_list"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    # Usar imshow en lugar de pcolormesh para una representación más eficiente
    ax.imshow(matriz, cmap=cmap, origin='upper')

    # Establecer la relación de aspecto para que las celdas sean cuadradas
    ax.set_aspect('equal')

    # Colocar las etiquetas del eje x en la parte superior
    ax.xaxis.tick_top()

    # Colocar las etiquetas del eje y en la parte izquierda
    ax.yaxis.tick_left()

    # plt.title("Iteracion {}".format(filename.split("_")[1].split(".")[0]))

    # Guardar la imagen
    # plt.savefig(filename)


def RepresentatePoints(matriz: np.ndarray, filename: str = "grafica.png") -> None:

    n = len(matriz)

    # Inicializamos listas para las coordenadas de los puntos
    x = []
    y = []

    x = [j for i in range(n) for j in range(len(matriz[i])) if matriz[i][j] == 1]
    y = [n - i - 1 for i in range(n) for j in range(len(matriz[i])) if matriz[i][j] == 1]

    # Crear el gráfico
    fig, ax = plt.subplots()

    plt.scatter(x, y, c='blue', marker='o')  # 'c' define el color y 'marker' define la forma de los puntos
    plt.title("Iteracion {}".format(filename.split("_")[1].split(".")[0]))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)  # Opcional: para mostrar la cuadrícula

    # Establecer la relación de aspecto para que las celdas sean cuadradas
    ax.set_aspect('equal')

    # Colocar las etiquetas del eje x en la parte superior
    ax.xaxis.tick_top()
    nrows, _ = matriz.shape

    # Configurar las marcas de los ejes
    major_ticks = np.arange(0, nrows, 1)
    minor_ticks = np.arange(0, nrows, 0.5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # Invertir el eje y para que el punto (0, 0) esté en la esquina superior izquierda
    ax.invert_yaxis()

    # Añado una leyenda fuera del gráfico
    plt.legend(['Trampas'], loc='center left', bbox_to_anchor=(1, 0.5))

    return None


def RepresentateStateOptAnto(matriz: np.ndarray, fig, ax, im=None, filename: str = "grafica.png") -> None:
    """
    Represent the state of a matrix as a colored plot.

    Parameters:
    - matriz (np.ndarray): The input matrix to be represented.
    - filename (str, optional): The name of the file to save the plot. Default is "grafica.png".

    Returns:
    None
    """
    # Crear una figura y un eje con plt.subplots()

    # Crear un mapa de colores personalizado
    colors = [
        (1, 1, 1),                  # Color para el valor 0 que representa que No hay trampa
        (0.478, 0.627, 0.870),      # Color para el valor 1 que representa que hay trampa (azul)
    ]
    if np.all(matriz == 1):
        colors = list(reversed(colors))

    if False:
        im.set_data(matriz)
    else:
        cmap_name = "my_list"
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

        # Usar imshow en lugar de pcolormesh para una representación más eficiente
        im = ax.imshow(matriz, cmap=cmap, origin='upper')

        # Establecer la relación de aspecto para que las celdas sean cuadradas
        ax.set_aspect('equal')

    # Colocar las etiquetas del eje x en la parte superior
    # ax.xaxis.tick_top()

    plt.title("Iteracion {}".format(filename.split("_")[1].split(".")[0]))

    return im


def plot_regions(Eje_x: int, Eje_y: int, regiones_pesos: list):
    """
    Plot the regions with privileged probability.

    Args:
        Eje_x (int): The size of the x-axis.
        Eje_y (int): The size of the y-axis.
        regiones_pesos (list): A list of tuples defining regions and their weights.
                               Each tuple should be ((x_start, x_end, y_start, y_end), weight).
    """
    fig, ax = plt.subplots()
    ax.set_xlim(0, Eje_y)
    ax.set_ylim(0, Eje_x)
    ax.invert_yaxis()

    # Draw grid
    for i in range(Eje_x):
        for j in range(Eje_y):
            rect = patches.Rectangle((j, i), 1, 1, edgecolor='grey', facecolor='white', fill=True)
            ax.add_patch(rect)

    # Highlight privileged regions
    for (x_start, x_end, y_start, y_end), weight in regiones_pesos:
        rect = patches.Rectangle((y_start, x_start), y_end - y_start, x_end - x_start,
                                 linewidth=4, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Add text for weight
        cx = (y_start + y_end) / 2
        cy = (x_start + x_end) / 2
        ax.text(cx, cy, f'w={weight}', color='red', ha='center', va='center', fontsize=8)

    plt.gca().set_aspect('equal', adjustable='box')

    filename = "Pruebas/Region_privilegiada.png"
    plt.savefig(filename)


if __name__ == "__main__":
    Longitud_Dispositivo = 10
    atom_size = 0.5

    Eje_x, Eje_y = round(Longitud_Dispositivo / atom_size), round(
        Longitud_Dispositivo / atom_size
    )
    num_trampas = 10

    # Crear una matriz de ceros de tamaño Eje_x x Eje_y
    InitialState = np.zeros((Eje_x, Eje_y), dtype=int)
    # Generar 5 posiciones aleatorias para los unos
    posiciones_unos = np.random.choice(Eje_x * Eje_y, num_trampas, replace=False)

    # Asignar el valor 1 a las posiciones seleccionadas
    for pos in posiciones_unos:
        fila, columna = divmod(pos, Eje_x)
        InitialState[fila, columna] = 1

    RepresentatePoints(InitialState, 'grafica0.png')
