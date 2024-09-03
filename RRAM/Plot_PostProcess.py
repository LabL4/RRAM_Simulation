import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

# Varias funciones para representar los datos obtenidos de la simulación


def Plot_panel(data_path: str, title: str = None) -> None:
    """
    Función que representa los datos obtenidos de la simulación en un panel con 4 subplots, acepta un archivo csv con los datos con la siguiente estructura:
        - La primera columna contiene la variable independiente
        - Las siguientes columnas contienen las variables dependientes

    Args:
    data_path: contiene la ruta del archivo de datos, se encuentra en la primera columna la variable independiente 
               y en las siguientes columnas las variables dependientes

    Returns:
        La figura con los 4 subplots representando los datos
    """

    # leo los datos desde el csv
    data = pd.read_csv(data_path)

    # Elimino la primera fila que son los nombres de las columnas
    data = data.values[1:, :]

    # Extraigo la variable independiente
    x = data[:, 0]

    # Extraigo las variables dependientes
    y = data[:, 1:]

    # Creo la figura que será un panel con 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 7))

    # Establezco el título del conjunto de figuras si se ha proporcionado uno
    if title is not None:
        fig.suptitle(title, fontsize=16)

    # Creo el primer subplot
    ax1.plot(x, y[:, 0])
    ax1.set_title('Velocidad')

    # añado una etiqueta al eje x
    ax1.set_xlabel('Tiempo [s]')
    ax1.set_ylabel('Velocidad [m/s]')

    # Creo el segundo subplot
    ax2.plot(x, y[:, 1])
    ax2.set_title('desplazamiento')

    # añado una etiqueta al eje x
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Desplazamiento [casillas]')

    # Creo el tercer subplot
    ax3.plot(x, y[:, 2])
    ax3.set_title('Probabilidad generacion')

    # añado una etiqueta al eje x
    ax3.set_xlabel('Tiempo [s]')

    # Creo el cuarto subplot
    ax4.plot(x, y[:, 3])
    ax4.set_title('Probabilidad recombinacion')

    # añado una etiqueta al eje x
    ax4.set_xlabel('Tiempo [s]')

    # Ajustamos el espacio entre los plots
    fig.tight_layout()

    # Ajusto el espacio para el título principal si se ha proporcionado uno
    if title is not None:
        fig.subplots_adjust(top=0.88)

    # TODO: Cambiar esto por si no le pongo titulo
    # Elimino la extensión del archivo
    data_path = (data_path.split('/')[1]).split('.')[0]
    partes = title.split(',')

    # Guardo la figura
    plt.savefig('Results/Panel_' + data_path + '_' + partes[0].split('=')
                [1].strip() + '-' + partes[1].split('=')[1].strip() + '.png')

    # Cierro la figura
    plt.close(fig)

    return None


def RepresentateALLState(state_matrix: np.ndarray, oxygen_matrix: np.ndarray, fig, ax, filename: str = "grafica.png") -> None:
    """
    Representates the state and oxygen matrices using a custom colormap and saves the plot as an image.

    Parameters:
    - state_matrix (np.ndarray): The state matrix to be represented.
    - oxygen_matrix (np.ndarray): The oxygen matrix to be represented.
    - fig: The figure object to plot on.
    - ax: The axes object to plot on.
    - filename (str): The name of the output image file (default: "grafica.png").

    Returns:
    None
    """

    # Create a custom color map for the first matrix
    colors1 = [
        (1, 1, 1),                 # Color for value 0 (white)
        (0.478, 0.627, 0.870),     # Color for value 1 (blue)
    ]

    # Create a custom color map for the second matrix
    colors2 = [
        (1, 1, 1),                 # Color for value 0 (white)
        (0.870, 0.478, 0.627),     # Color for value 1 (red)
    ]

    cmap1 = LinearSegmentedColormap.from_list("cmap1", colors1, N=2)
    cmap2 = LinearSegmentedColormap.from_list("cmap2", colors2, N=2)

    # Use imshow to represent the configuration matrix
    ax.imshow(state_matrix, cmap=cmap1, origin='upper', alpha=0.85)

    # Use imshow to represent the oxygen matrix
    ax.imshow(oxygen_matrix, cmap=cmap2, origin='upper', alpha=0.45)

    # Set the aspect ratio to make cells square
    ax.set_aspect('equal')

    # Set the x-axis labels at the top
    # ax.xaxis.tick_top()

    # plt.title("Iteration {}".format(filename.split("_")[1].split(".")[0]))

    # Close the figure and save it to a file
    plt.savefig(filename)
    plt.close(fig)

    return None


def Plot_2panel(data_path: str, col_indices_x: list, col_indices_y: list, save_path: str, global_tittle: str = None,
                titles: list = None, eje_x: list = None, eje_y: list = None, log_scale: list = [None, None]) -> None:
    """Plot_2panel Representate the data from a CSV file in a 2-panel plot.

    Args:
        data_path (str): the path to the CSV file with the data.
        col_indices_x (list): the indices of the columns to use as the x-axis.
        col_indices_y (list): the indices of the columns to use as the y-axis.
        global_tittle (str, optional): the tittle of the all figure. Defaults to None.
        titles (list, optional): the titles of each figure. Defaults to None.
        eje_x (list, optional): the x axis names. Defaults to None.
        eje_y (list, optional): the y axis names. Defaults to None.
        log_scale (list, optional): activate or not log plot in each plot, 'x' for x axis 'y' for y axis an 'both' for xy axis. Defaults to [None, None].
    """
    # Leer los datos desde el CSV
    data = pd.read_csv(data_path)

    # for i, column in enumerate(data.columns):
    #     print(f"Columna {i}: {column}")

    print("\n")
    x1, x2 = data.iloc[:, col_indices_x[0]], data.iloc[:, col_indices_x[1]]
    y1, y2 = data.iloc[:, col_indices_y[0]], data.iloc[:, col_indices_y[1]]

    # # Imprimir los nombres de las columnas
    # print("x1 pertenece a la columna:", data.columns[col_indices_x[0]])
    # print("y1 pertenece a la columna:", data.columns[col_indices_y[0]])
    # print("x2 pertenece a la columna:", data.columns[col_indices_x[1]])
    # print("y2 pertenece a la columna:", data.columns[col_indices_y[1]])

    # Crear la figura y los subplots
    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    # Asignar datos y títulos/etiquetas si se proporcionan
    for i, (ax, x, y) in enumerate(zip(axes, [x1, x2], [y1, y2])):
        ax.scatter(x, y, s=5)
        if titles and len(titles) > i:
            ax.set_title(titles[i])
        if eje_x and len(eje_x) > i:
            ax.set_xlabel(eje_x[i])
        else:
            ax.set_xlabel(data.columns[col_indices_x[i]])
        if eje_y and len(eje_y) > i:
            ax.set_ylabel(eje_y[i])
        else:
            ax.set_ylabel(data.columns[col_indices_y[i]])
        if log_scale and len(log_scale) > i:
            if log_scale[i] == 'x':
                ax.set_xscale('log')
            elif log_scale[i] == 'y':
                ax.set_yscale('log')
            elif log_scale[i] == 'both':
                ax.set_xscale('log')
                ax.set_yscale('log')

    # Título general
    if global_tittle:
        fig.suptitle(global_tittle, fontsize=16)
        fig.subplots_adjust(top=0.88)

    # Ajustar diseño y guardar la figura
    fig.tight_layout()

    plt.savefig(f'{save_path}')
