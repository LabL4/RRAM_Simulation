# Description: This file contains the code to generate the mesh of the RRAM.
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap


def RepresentateState(matriz: np.ndarray, filename: str = "grafica.png"):
    """
    Represent the state of a matrix as a colored plot.

    Parameters:
    - matriz (np.ndarray): The input matrix to be represented.

    Returns:
    None
    """
    nrows, ncols = matriz.shape

    x = np.arange(ncols)
    y = np.arange(nrows)

    fig, ax = plt.subplots()

    # Indico qué valores van en las subdivisiones
    colors = [
        (1, 1, 1),  # Color para el valor 0
        (0.478, 0.627, 0.870),  # Color para el valor 1
    ]
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
    major_ticks = np.arange(0, 10, 10)
    minor_ticks = np.arange(0, 10, 5)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # Establecer relación de aspecto cuadrada
    ax.set_aspect("equal")

    # # Añadir líneas gruesas horizontales al inicio y al final
    # ax.axvline(x=-0.5, color="black", linewidth=4)  # Línea al inicio
    # ax.axvline(x=nrows + 0.5, color="black", linewidth=4)  # Línea al final

    # Guardar la imagen
    plt.savefig(filename)


if __name__ == "__main__":
    Longitud_Dispositivo = 10
    Atom_size = 0.5

    Eje_x, Eje_y = round(Longitud_Dispositivo / Atom_size), round(
        Longitud_Dispositivo / Atom_size
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

    RepresentateState(InitialState)
