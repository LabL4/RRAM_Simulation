import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import markers

# import pandas as pd
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
# from RRAM import Montecarlo

import os
# region Configuración del plot


def config_ax(ax):
    ax.grid(which="major", color="#DDDDDD", linewidth=0.8, zorder=-1)
    ax.grid(which="minor", color="#DEDEDE", linestyle=":", linewidth=0.5, zorder=-1)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def setup_plt(plt, latex=True, scaling=1):
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": latex,
            "font.family": "fourier",
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                ]
            ),
        }
    )

    SMALL_SIZE = 8 * scaling
    MEDIUM_SIZE = 10 * scaling
    BIGGER_SIZE = 11 * scaling

    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=SMALL_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=SMALL_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)
    plt.rc("axes", titlesize=BIGGER_SIZE * 1.05)


setup_plt(plt, latex=True, scaling=2)


def RepresentateState(
    matriz: np.ndarray,
    voltaje: float,
    filename: str = None,  # type: ignore
    color=(0.9647, 0.1725, 0.3059),
) -> None:  # type: ignore
    """
    Representa el estado de una matriz con un estilo gráfico personalizado.

    Parámetros:
    - matriz (np.ndarray): Matriz a representar.
    - k (int): Índice de iteración para el voltaje.
    - paso_voltaje (float): Incremento de voltaje por iteración.
    - filename (str, opcional): Nombre del archivo para guardar la gráfica.
    - color (tuple, opcional): Color de las vacantes.

    Retorna:
    - None
    """

    nrows, ncols = matriz.shape
    x = np.linspace(0, 10, ncols)  # Escala real de 10 nm en eje X
    y = np.linspace(0, 10, nrows)  # Escala real de 10 nm en eje Y

    fig, ax = plt.subplots(figsize=(12, 9))

    # Crear mapa de colores
    colors = [(1, 1, 1), color]  # Blanco (0) y Color dado (1)
    if np.all(matriz == 1):
        colors.reverse()

    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)

    # Graficar la matriz
    c = ax.pcolormesh(
        x, y, matriz, shading="nearest", vmin=matriz.min(), vmax=matriz.max(), cmap=cmap
    )

    # Configuración de electrodos con mayor altura
    electrode_width = 0.2  # Se mantiene el ancho en X
    electrode_height = 12  # Se extienden en Y (de -1 a 11)
    electrode_color = "gray"  # Color de los electrodos

    left_electrode = patches.Rectangle(
        (-0.3, -0.5),  # Posición en X, Y (más abajo)
        electrode_width,
        electrode_height,  # Ancho y Alto aumentados
        color=electrode_color,
    )

    right_electrode = patches.Rectangle(
        (10.13, -1),  # Posición en X, Y (más abajo)
        electrode_width,
        electrode_height,  # Ancho y Alto aumentados
        color=electrode_color,
    )

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # Aplicar configuración de ejes
    config_ax(ax)

    # Configurar etiquetas y título
    ax.set_xticks(np.arange(0, 11, 2))  # 🔹 Ticks cada 2 nm en X
    ax.set_yticks(np.arange(0, 11, 2))  # 🔹 Ticks cada 2 nm en Y
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")
    ax.set_title(rf"V = {voltaje} (V)", pad=20)

    # Ajustar formato visual
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Ajustar límites del eje X y Y para que los electrodos sean visibles
    ax.set_xlim(-0.3, 10.33)
    ax.set_ylim(-0, 10)  # 🔥 Extiende el gráfico en Y para acomodar los electrodos

    # Aumentar margen superior para más espacio en el título
    plt.subplots_adjust(top=0.85)

    # Guardar si se especifica un archivo
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    # Mostrar gráfico
    plt.close(fig)

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
        (1, 1, 1),  # Color para el valor 0 que representa que No hay trampa
        (
            0.478,
            0.627,
            0.870,
        ),  # Color para el valor 1 que representa que hay trampa (azul)
    ]
    if np.all(matriz == 1):
        colors = list(reversed(colors))

    cmap_name = "my_list"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

    # Usar imshow en lugar de pcolormesh para una representación más eficiente
    ax.imshow(matriz, cmap=cmap, origin="upper")

    # Establecer la relación de aspecto para que las celdas sean cuadradas
    ax.set_aspect("equal")

    # Colocar las etiquetas del eje x en la parte superior
    ax.xaxis.tick_top()

    # Colocar las etiquetas del eje y en la parte izquierda
    ax.yaxis.tick_left()

    # plt.title("Iteracion {}".format(filename.split("_")[1].split(".")[0]))

    # Guardar la imagen
    plt.savefig(filename)


def RepresentateState_parall(
    matriz: np.ndarray,
    fig,
    ax,
    im=None,
    color=(0.478, 0.627, 0.870),
    filename: str = "grafica.png",
) -> None:
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
        (1, 1, 1),  # Color para el valor 0 que representa que No hay trampa
        color,  # Color para el valor 1 que representa que hay trampa (azul por defeto)
    ]
    if np.all(matriz == 1):
        colors = list(reversed(colors))

    if False:
        im.set_data(matriz)
    else:
        cmap_name = "my_list"
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

        # Usar imshow en lugar de pcolormesh para una representación más eficiente
        im = ax.imshow(matriz, cmap=cmap, origin="upper")

        # Establecer la relación de aspecto para que las celdas sean cuadradas
        ax.set_aspect("equal")

    # Colocar las etiquetas del eje x en la parte superior
    # ax.xaxis.tick_top()

    # sim_parmtrs = Montecarlo.read_csv_to_dic("Init_data/simulation_parameters.csv")
    # num_simulation = 0

    # num_pasos = int(sim_parmtrs[num_simulation]['num_pasos'])
    # voltaje_final = float(sim_parmtrs[num_simulation]['voltaje_final'])

    # vector_ddp = np.linspace(0, voltaje_final, num_pasos + 1)
    # # vector_ddp = np.linspace(voltaje_final, 0, num_pasos + 1)
    iteracion = int(filename.split("_")[1].split(".")[0])
    # potencial = vector_ddp[iteracion-1]

    plt.title(f"iteracion {iteracion}")
    # plt.title(f"potencial: {potencial:.4f} V, iteracion {iteracion}")

    # plt.title("Iteracion {}".format(filename.split("_")[1].split(".")[0]))

    return im


def RepresentateStateOxygen(
    matriz: np.ndarray, fig, ax, im=None, filename: str = "grafica.png"
) -> None:
    """
    Represent the state of a matrix as a colored plot. Es la misma funcion que arriba solo que pinta
    en rojo los oxigenos (solo los soxigenos pinta) para diferenciarlos de las trampas

    Parameters:
    - matriz (np.ndarray): The input matrix to be represented.
    - filename (str, optional): The name of the file to save the plot. Default is "grafica.png".

    Returns:
    None
    """
    # Crear una figura y un eje con plt.subplots()

    # Crear un mapa de colores personalizado
    colors = [
        (1, 1, 1),  # Color para el valor 0 que representa que No hay trampa
        (
            0.9922,
            0.2157,
            0.2157,
        ),  # Color para el valor 1 que representa que hay oxigeno (rojo)
    ]
    if np.all(matriz == 1):
        colors = list(reversed(colors))

    if False:
        im.set_data(matriz)
    else:
        cmap_name = "my_list"
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)

        # Usar imshow en lugar de pcolormesh para una representación más eficiente
        im = ax.imshow(matriz, cmap=cmap, origin="upper")

        # Establecer la relación de aspecto para que las celdas sean cuadradas
        ax.set_aspect("equal")

    # Colocar las etiquetas del eje x en la parte superior
    # ax.xaxis.tick_top()

    plt.title("Iteracion {}".format(filename.split("_")[1].split(".")[0]))

    return im


def plot_regions(Eje_x: int, Eje_y: int, regiones_pesos: list, filename: str) -> None:
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
            rect = patches.Rectangle(
                (j, i), 1, 1, edgecolor="grey", facecolor="white", fill=True
            )
            ax.add_patch(rect)

    # Highlight privileged regions
    for (x_start, x_end, y_start, y_end), weight in regiones_pesos:
        rect = patches.Rectangle(
            (y_start, x_start),
            y_end - y_start,
            x_end - x_start,
            linewidth=4,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        # Add text for weight
        cx = (y_start + y_end) / 2
        cy = (x_start + x_end) / 2
        ax.text(
            cx, cy, f"w={weight}", color="red", ha="center", va="center", fontsize=8
        )

    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(filename)


def plot_privileged_regions(
    Eje_x: int, Eje_y: int, regiones_pesos: list, filename: str
) -> None:
    """
    Plot the privileged regions on a grid.

    Args:
        Eje_x (int): The size of the x-axis.
        Eje_y (int): The size of the y-axis.
        regiones_pesos (list): A list of tuples defining regions and their weights.
                               Each tuple should be ((x_start, x_end, y_start, y_end), weight).
        filename (str): The name of the file to save the plot.
    """
    fig, ax = plt.subplots()

    # Verificar que los límites no sean iguales
    if Eje_y == 0:
        Eje_y = 1
    if Eje_x == 0:
        Eje_x = 1

    ax.set_xlim(0, Eje_y)
    ax.set_ylim(0, Eje_x)
    ax.invert_yaxis()

    # Draw grid
    for i in range(Eje_x):
        for j in range(Eje_y):
            rect = patches.Rectangle(
                (j, i), 1, 1, edgecolor="grey", facecolor="white", fill=True
            )
            ax.add_patch(rect)

    # Highlight privileged regions
    for (x_start, x_end, y_start, y_end), weight in regiones_pesos:
        rect = patches.Rectangle(
            (y_start, x_start),
            y_end - y_start,
            x_end - x_start,
            linewidth=4,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        # Add text for weight
        cx = (y_start + y_end) / 2
        cy = (x_start + x_end) / 2
        ax.text(
            cx, cy, f"w={weight}", color="red", ha="center", va="center", fontsize=8
        )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(filename)
    plt.close(fig)


def RepresentateTwoStates(
    matriz1: np.ndarray,
    matriz2: np.ndarray,
    voltage: float,
    filename: str = None,  # type: ignore
) -> None:
    """
    Representa el estado de dos matrices con un estilo gráfico personalizado en el mismo plot.

    Parámetros:
    - matriz1 (np.ndarray): Primera matriz a representar (color rojo).
    - matriz2 (np.ndarray): Segunda matriz a representar (color azul).
    - k (int): Factor para calcular el voltaje.
    - paso_voltaje (float): Paso de voltaje.
    - filename (str, opcional): Nombre del archivo para guardar la gráfica.

    Retorna:
    - None
    """

    nrows, ncols = matriz1.shape
    x = np.linspace(0, 10, ncols)  # Escala real de 10 nm en eje X
    y = np.linspace(0, 10, nrows)  # Escala real de 10 nm en eje Y

    fig, ax = plt.subplots(figsize=(12, 9))  # Tamaño de la figura ajustado

    # Crear mapas de colores para cada matriz
    cmap1 = LinearSegmentedColormap.from_list(
        "cmap1", [(1, 1, 1), (0.9647, 0.1725, 0.3059)], N=2
    )  # Rojo
    cmap2 = LinearSegmentedColormap.from_list(
        "cmap2", [(1, 1, 1), (0.2314, 0.2275, 0.9647)], N=2
    )  # Azul

    # Graficar la primera matriz
    ax.pcolormesh(
        x,
        y,
        matriz1,
        shading="nearest",
        vmin=matriz1.min(),
        vmax=matriz1.max(),
        cmap=cmap1,
    )

    # Graficar la segunda matriz
    ax.pcolormesh(
        x,
        y,
        matriz2,
        shading="nearest",
        vmin=matriz2.min(),
        vmax=matriz2.max(),
        cmap=cmap2,
        alpha=0.5,  # Transparencia para superposición
    )

    # Configuración de electrodos con mayor altura
    electrode_width = 0.2  # Se mantiene el ancho en X
    electrode_height = 12  # 🔥 Se extienden en Y (de -1 a 11)
    electrode_color = "gray"  # Color de los electrodos

    left_electrode = patches.Rectangle(
        (-0.3, -0.5),  # Posición en X, Y (más abajo)
        electrode_width,
        electrode_height,  # Ancho y Alto aumentados
        color=electrode_color,
    )

    right_electrode = patches.Rectangle(
        (10.13, -1),  # Posición en X, Y (más abajo)
        electrode_width,
        electrode_height,  # Ancho y Alto aumentados
        color=electrode_color,
    )

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)
    # Aplicar configuración de ejes
    config_ax(ax)

    # Configurar etiquetas y título
    ax.set_xticks(np.arange(0, 11, 2))  # Ticks cada 2 nm en X
    ax.set_yticks(np.arange(0, 11, 2))  # Ticks cada 2 nm en Y
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")
    ax.set_title(rf"V = {voltage} (V)", pad=20)

    # Ajustar formato visual
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Ajustar límites del eje X y Y para que los electrodos sean visibles
    ax.set_xlim(-0.3, 10.33)
    ax.set_ylim(-0, 10)  # 🔥 Extiende el gráfico en Y para acomodar los electrodos

    # Aumentar margen superior para más espacio en el título
    plt.subplots_adjust(top=0.85)

    # Guardar si se especifica un archivo
    if filename:
        plt.savefig(filename, bbox_inches="tight")

    # Mostrar gráfico
    plt.close(fig)

    return None


def plot_IV(
    v_set,
    i_set,
    v_reset,
    i_reset,
    num_simulation,
    titulo_figura="I-V Characteristics",
    figures_path="C:/Users/jimdo/Documents/GitHub/RRAM_Simulation/Results/Figures",
):
    # figures_path="C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Results/Figures",
    """
    Plots the I-V characteristics of a device.
    Parameters:
        v_set (list): List of SET voltages.
        i_set (list): List of SET currents.
        v_reset (list): List of RESET voltages.
        i_reset (list): List of RESET currents.
        num_simulation (int): Simulation number for saving the figure.
        titulo_figura (str): Title of the figure.
        figures_path (str): Path to save the figure.
    """

    figures_path = os.getcwd() + "/Results/Figures"

    # Configuración de la figura
    setup_plt(plt, latex=True, scaling=2)

    fig, axes = plt.subplots(figsize=(12, 9))
    config_ax(axes)

    axes.set_xlabel("Voltage [V]")
    axes.set_ylabel("Current [A]")
    axes.set_yscale("log")
    axes.set_title(titulo_figura, pad=20)

    # Scatter de SET y RESET
    axes.scatter(
        v_set,
        i_set,
        color="red",
        s=15,
        marker=markers.MarkerStyle("o"),
        facecolors="white",
        label="SET",
    )
    axes.scatter(
        v_reset,
        i_reset,
        color="red",
        s=15,
        marker=markers.MarkerStyle("s"),
        facecolors="white",
        label="RESET",
    )

    # Ruta de los datos experimentales
    # ruta_archivo_set = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt'
    # ruta_archivo_reset = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt'
    ruta_archivo_set = (
        os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt"
    )
    ruta_archivo_reset = (
        os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt"
    )

    # Cargar datos experimentales
    data_set = np.loadtxt(ruta_archivo_set)
    data_reset = np.loadtxt(ruta_archivo_reset)

    x_set = data_set[:, 0]
    y_set = data_set[:, 1]
    x_reset = data_reset[:, 0]
    y_reset = abs(data_reset[:, 1])

    # Curvas experimentales
    axes.plot(x_set, y_set, "black", label="Set experimental", linewidth=2.5)
    axes.plot(x_reset, y_reset, "black", label="Reset experimental", linewidth=2.5)

    # Leyenda ajustada en la parte inferior izquierda
    axes.legend(
        labelspacing=0.3,
        handletextpad=0.2,
        handlelength=1.0,
        borderaxespad=0.2,
        loc="lower left",
    )

    # Guardar figura
    fig.savefig(
        figures_path + f"/I-V_{num_simulation + 1}.png", bbox_inches="tight", dpi=300
    )
    plt.close(fig)  # Cierra para liberar memoria


def plot_IV_marcado(
    v_set,
    i_set,
    v_reset,
    i_reset,
    num_simulation,
    lista_puntos,
    desplazamiento,
    titulo_figura="I-V Characteristics",
    figures_path="C:/Users/jimdo/Documents/GitHub/RRAM_Simulation/Results/Figures",
):
    # figures_path="C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Results/Figures",
    """
    Plots the I-V characteristics of a device.
    Parameters:
        v_set (list): List of SET voltages.
        i_set (list): List of SET currents.
        v_reset (list): List of RESET voltages.
        i_reset (list): List of RESET currents.
        num_simulation (int): Simulation number for saving the figure.
        titulo_figura (str): Title of the figure.
        figures_path (str): Path to save the figure.
    """

    figures_path = os.getcwd() + "/Results/Figures"

    # Configuración de la figura
    setup_plt(plt, latex=True, scaling=2)

    fig, axes = plt.subplots(figsize=(12, 9))
    config_ax(axes)

    axes.set_xlabel("Voltage [V]")
    axes.set_ylabel("Current [A]")
    axes.set_yscale("log")
    axes.set_title(titulo_figura, pad=20)

    # Scatter de SET y RESET
    axes.scatter(
        v_set,
        i_set,
        color="red",
        s=15,
        marker=markers.MarkerStyle("o"),
        facecolors="white",
        label="SET",
    )
    axes.scatter(
        v_reset,
        i_reset,
        color="red",
        s=15,
        marker=markers.MarkerStyle("s"),
        facecolors="white",
        label="RESET",
    )

    # Ruta de los datos experimentales
    # ruta_archivo_set = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt'
    # ruta_archivo_reset = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt'
    ruta_archivo_set = (
        os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt"
    )
    ruta_archivo_reset = (
        os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt"
    )

    # Cargar datos experimentales
    data_set = np.loadtxt(ruta_archivo_set)
    data_reset = np.loadtxt(ruta_archivo_reset)

    x_set = data_set[:, 0]
    y_set = data_set[:, 1]
    x_reset = data_reset[:, 0]
    y_reset = abs(data_reset[:, 1])

    (x_0, y_0) = next(iter(lista_puntos.values()))
    print("Punto de referencia (0,0): ", (x_0, y_0))
    axes.scatter(
        1e-6,
        1e-6,
        color="blue",
        s=80,
        marker=markers.MarkerStyle("D"),
        zorder=10,
    )

    # Curvas experimentales
    axes.plot(x_set, y_set, "black", label="Set experimental", linewidth=2.5)
    axes.plot(x_reset, y_reset, "black", label="Reset experimental", linewidth=2.5)

    for label, (xp, yp) in lista_puntos.items():
        dx, factor_y = desplazamiento.get(
            label, (0.02, 1.0)
        )  # 1.0 = sin desplazamiento en y
        print(
            label,
            "puntos: ",
            (xp, yp),
            (dx, factor_y),
            "puntos finales: ",
            (xp + dx, yp * factor_y),
        )
        axes.scatter(
            xp, yp, color="blue", s=80, marker=markers.MarkerStyle("D"), zorder=10
        )
        axes.text(
            xp + dx,  # Usar la posición calculada en x
            max(yp * factor_y, 1e-6),  # Usar la posición calculada en y con un mínimo
            label,
            fontsize=22,  # Reducir el tamaño de fuente
            verticalalignment="bottom",
            horizontalalignment="left",
            zorder=10,
        )

    # Leyenda ajustada en la parte inferior izquierda
    axes.legend(
        labelspacing=0.3,
        handletextpad=0.2,
        handlelength=1.0,
        borderaxespad=0.2,
        loc="lower left",
    )

    # Guardar figura
    fig.savefig(
        figures_path + f"/I-V_marcado_{num_simulation + 1}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)  # Cierra para liberar memoria


if __name__ == "__main__":
    Longitud_Dispositivo = 10
    atom_size = 0.5

    Eje_x, Eje_y = (
        round(Longitud_Dispositivo / atom_size),
        round(Longitud_Dispositivo / atom_size),
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
