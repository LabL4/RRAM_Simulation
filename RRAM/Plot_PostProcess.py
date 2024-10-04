import numpy as np
from cv2 import log
import pandas as pd
from turtle import setup
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap

# Varias funciones para representar los datos obtenidos de la simulación


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


def Plot_paneles(data_path: str, col_indices_x: list, col_indices_y: list, save_path: str, global_tittle: str = None,
                 titles: list = None, eje_x: list = None, eje_y: list = None, log_scale: list = None) -> None:
    """Plot_paneles Representa los datos de un archivo CSV en un máximo de 4 paneles.

    Args:
        data_path (str): la ruta al archivo CSV con los datos.
        col_indices_x (list): los índices de las columnas a usar como eje x.
        col_indices_y (list): los índices de las columnas a usar como eje y.
        global_tittle (str, optional): el título de toda la figura. Defaults a None.
        titles (list, optional): los títulos de cada subgráfico. Defaults a None.
        eje_x (list, optional): los nombres de los ejes x. Defaults a None.
        eje_y (list, optional): los nombres de los ejes y. Defaults a None.
        log_scale (list, optional): activar o no escala logarítmica en cada gráfico, 'x' para el eje x, 'y' para el eje y, 'both' para ambos ejes. Defaults a None.
    """

    # Limitar el número máximo de gráficos a 4
    max_plots = 4
    num_plots = min(len(col_indices_x), len(col_indices_y), max_plots)

    if num_plots == 0:
        print("No hay datos suficientes para graficar.")
        return

    # Leer los datos desde el archivo CSV
    data = pd.read_csv(data_path)

    # Determinar la disposición de los subplots
    if num_plots == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
        axes = [axes]  # Convertir en lista para consistencia
    elif num_plots == 2:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 8))  # 2 filas y 1 columna
    else:
        fig, axes = plt.subplots(nrows=(num_plots + 1) // 2, ncols=2)  # Hasta 4 gráficos, 2x2 como máximo

    # Si hay más de un gráfico, ravel para convertir el array en una lista
    if num_plots > 1:
        axes = axes.ravel()

    # Asignar los datos y realizar las gráficas
    for i in range(num_plots):
        x = data.iloc[:, col_indices_x[i]]
        y = data.iloc[:, col_indices_y[i]]

        ax = axes[i]
        config_ax(ax)
        ax.scatter(x, y, s=5, zorder=10)

        # Asignar títulos y etiquetas de ejes si se proporcionan
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

        # Escala logarítmica si se solicita
        if log_scale and len(log_scale) > i:
            if log_scale[i] == 'x':
                ax.set_xscale('log')
            elif log_scale[i] == 'y':
                ax.set_yscale('log')
            elif log_scale[i] == 'both':
                ax.set_xscale('log')
                ax.set_yscale('log')

    # Eliminar subplots vacíos si el número de gráficos es menor a 4
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Título global
    if global_tittle:
        fig.suptitle(global_tittle, fontsize=16)
        fig.subplots_adjust(top=0.88)

    # Ajustar el diseño y guardar la figura
    fig.tight_layout()

    # Guardar la figura
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    # plt.close(fig)


def plot_DifAxes(data_path: str,
                 col_indices_x: int,
                 col_indices_y: list,
                 save_path: str,
                 global_tittle: str = ' '
                 ) -> None:

    # leo los datos desde el csv
    data = pd.read_csv(data_path)

    # Extraigo la variable independiente
    x1 = data.iloc[:, col_indices_x]

    # Extraigo las variables dependientes
    y1 = data.iloc[:, col_indices_y[0]]

    # Extraigo la variable independiente
    x2 = data.iloc[:, col_indices_x]

    # Extraigo las variables dependientes
    y2 = data.iloc[:, col_indices_y[1]]

    print(data.columns[col_indices_x])
    print(data.columns[col_indices_y[0]])
    print(data.columns[col_indices_y[1]])

    fig, axes = plt.subplots()
    config_ax(axes)

    axes.set_xlabel(data.columns[col_indices_x])
    axes.set_ylabel(data.columns[col_indices_y[0]], color='blue')

    axes.set_title(global_tittle, fontsize=16, pad=15)

    axes.scatter(x1, y1, s=1.5, color='blue')

    twin_axes = axes.twinx()
    twin_axes.scatter(x2, y2, s=1.5, color='r')
    twin_axes.set_ylabel(data.columns[col_indices_y[1]], color='r')

    plt.show()
    fig.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_both(data_path: str,
              col_indices_x: int,
              col_indices_y: list,
              y_label: str,
              save_path: str,
              global_tittle: str = ' ',
              log_scale: list = None
              ) -> None:

    # leo los datos desde el csv
    data = pd.read_csv(data_path)

    # Extraigo la variable independiente
    x1 = data.iloc[:, col_indices_x]

    # Extraigo las variables dependientes
    y1 = data.iloc[:, col_indices_y[0]]

    # Extraigo la variable independiente
    x2 = data.iloc[:, col_indices_x]

    # Extraigo las variables dependientes
    y2 = data.iloc[:, col_indices_y[1]]

    # print(data.columns[col_indices_x])
    # print(data.columns[col_indices_y[0]])
    # print(data.columns[col_indices_y[1]])

    fig, axes = plt.subplots()
    config_ax(axes)

    axes.set_xlabel(data.columns[col_indices_x])
    axes.set_ylabel(y_label)

    # Escala logarítmica si se solicita
    if log_scale and len(log_scale) > 0:
        if log_scale[0] == 'x':
            axes.set_xscale('log')
        elif log_scale[0] == 'y':
            axes.set_yscale('log')
        elif log_scale[0] == 'both':
            axes.set_xscale('log')
            axes.set_yscale('log')

    axes.set_title(global_tittle, fontsize=18, pad=15)

    axes.scatter(x1, y1, s=1.5)
    axes.scatter(x2, y2, s=1.5)

    plt.show()
    fig.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_simple(data_path: str,
                col_indices_x: int,
                col_indices_y: int,
                save_path: str,
                global_tittle: str = ' ',
                x_label: str = ' ',
                y_label: str = ' ',
                log_scale: list = None
                ) -> None:

    # leo los datos desde el csv
    data = pd.read_csv(data_path)

    # Extraigo la variable independiente
    x1 = data.iloc[:, col_indices_x]

    # Extraigo las variables dependientes
    y1 = data.iloc[:, col_indices_y]

    # print(data.columns[col_indices_x])
    # print(data.columns[col_indices_y[0]])
    # print(data.columns[col_indices_y[1]])

    fig, axes = plt.subplots()
    config_ax(axes)

    axes.set_xlabel(x_label)
    if x_label == ' ':  # Si no se proporciona etiqueta para el eje x, usar el nombre de la columna
        axes.set_xlabel(data.columns[col_indices_x])
    else:
        axes.set_xlabel(x_label)

    if y_label == ' ':  # Si no se proporciona etiqueta para el eje y, usar el nombre de la columna
        axes.set_ylabel(data.columns[col_indices_y])
    else:
        axes.set_ylabel(y_label)

    # Escala logarítmica si se solicita
    if log_scale and len(log_scale) > 0:
        if log_scale[0] == 'x':
            axes.set_xscale('log')
        elif log_scale[0] == 'y':
            axes.set_yscale('log')
        elif log_scale[0] == 'both':
            axes.set_xscale('log')
            axes.set_yscale('log')

    axes.set_title(global_tittle, fontsize=18, pad=15)

    axes.scatter(x1, y1, s=1.5)

    plt.show()
    fig.savefig(save_path + '.pdf', bbox_inches='tight')
    plt.close(fig)


def config_ax(ax):
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=-1)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax.grid(which='minor', color='#DEDEDE', linestyle=':', linewidth=0.5, zorder=-1)
    # Make the minor ticks and gridlines show.
    ax.minorticks_on()

    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)


def setup_plt(plt, latex=True, scaling=1):

    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": latex,
            "font.family": "fourier",
            "text.latex.preamble": "\n".join([  # plots will use this preamble
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{siunitx}",
            ])
        }
    )

    SMALL_SIZE = 8 * scaling
    MEDIUM_SIZE = 10 * scaling
    BIGGER_SIZE = 12 * scaling

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


setup_plt(plt, latex=True, scaling=1.5)
