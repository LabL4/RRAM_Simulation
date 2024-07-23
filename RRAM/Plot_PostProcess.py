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
    ax2.set_ylabel('Desplazamiento [m]')

    # Creo el tercer subplot
    ax3.plot(x, y[:, 2])
    ax3.set_title('Probabilidad generacion')

    # añado una etiqueta al eje x
    ax3.set_xlabel('Tiempo [s]')

    # Creo el cuarto subplot
    ax4.plot(x, y[:, 3])
    ax4.set_title('sinh')

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
