import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib import markers

# import pandas as pd
import numpy as np

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
# from RRAM import Montecarlo

import os
# region Configuración del plot


def config_ax(ax):
    ax.grid(which="major", color="#DDDDDD", linewidth=0.8, zorder=-1)
    ax.grid(which="minor", color="#DEDEDE", linestyle=":", linewidth=0.5, zorder=-1)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def config_ax_IV(ax):
    # ax.grid(which="major", color="#DDDDDD", linewidth=0.8, zorder=-1)
    # ax.grid(which="minor", color="#DEDEDE", linestyle=":", linewidth=0.5, zorder=-1)
    ax.minorticks_on()
    # ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def config_ax_state(ax):
    # ax.grid(which="major", color="#DDDDDD", linewidth=0.8, zorder=-1)
    # ax.grid(which="minor", color="#DEDEDE", linestyle=":", linewidth=0.5, zorder=-1)
    ax.minorticks_off()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def setup_paper_plt(plt, latex=True, scaling: float = 1):
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": latex,
            "font.family": "mathpazo",
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                    r"\usepackage{physics}",
                ]
            ),
        }
    )

    # MEDIUM_SIZE = 10 * scaling
    BIGGER_SIZE = 11 * scaling
    BIGGEST_SIZE = 14 * scaling

    plt.rc("font", size=BIGGER_SIZE)
    plt.rc("axes", titlesize=BIGGEST_SIZE)
    plt.rc("axes", labelsize=BIGGEST_SIZE)
    plt.rc("xtick", labelsize=BIGGEST_SIZE)
    plt.rc("ytick", labelsize=BIGGEST_SIZE)
    plt.rc("legend", fontsize=BIGGER_SIZE)
    plt.rc("figure", titlesize=BIGGEST_SIZE)


def setup_plt(plt, latex=True, scaling=1):
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": latex,
            "font.family": "mathpazo",
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                    r"\usepackage{physics}",
                ]
            ),
        }
    )

    SMALL_SIZE = 8 * scaling
    MEDIUM_SIZE = 10 * scaling
    BIGGER_SIZE = 12 * scaling

    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=SMALL_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=SMALL_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)
    plt.rc("axes", titlesize=BIGGER_SIZE * 1.5)


setup_paper_plt(plt, latex=True, scaling=3)


# # (0.9647, 0.1725, 0.3059) color rojo original
# def RepresentateState(
#     matriz: np.ndarray,
#     voltaje: float,
#     filename: str = None,  # type: ignore
#     color=(0.0000, 0.0000, 0.5451),
#     guardar_png: bool = False,
# ) -> None:  # type: ignore
#     """
#     Representa el estado de una matriz con un estilo gráfico personalizado.

#     Parámetros:
#     - matriz (np.ndarray): Matriz a representar.
#     - k (int): Índice de iteración para el voltaje.
#     - paso_voltaje (float): Incremento de voltaje por iteración.
#     - filename (str, opcional): Nombre del archivo para guardar la gráfica.
#     - color (tuple, opcional): Color de las vacantes.

#     Retorna:
#     - None
#     """

#     nrows, ncols = matriz.shape
#     # x = np.linspace(0, 10, ncols)  # Escala real de 10 nm en eje X
#     # y = np.linspace(0, 10, nrows)  # Escala real de 10 nm en eje Y

#     x = np.linspace(0, 10, ncols)  # Escala real de 10 nm en eje X
#     y = np.linspace(0, 10, nrows)  # Escala real de 10 nm en eje Y

#     fig, ax = plt.subplots(figsize=(12, 9))

#     setup_paper_plt(plt, latex=True, scaling=3)
#     config_ax_state(ax)
#     # CUSTOM_SIZE = 32

#     # plt.rc("axes", labelsize=CUSTOM_SIZE)
#     # plt.rc("xtick", labelsize=CUSTOM_SIZE)
#     # plt.rc("ytick", labelsize=CUSTOM_SIZE)
#     # Desactivar minorticks para evitar sobrecarga visual

#     # Crear mapa de colores
#     colors = [(1, 1, 1), color]  # Blanco (0) y Color dado (1)
#     if np.all(matriz == 1):
#         colors.reverse()

#     cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)

#     # Graficar la matriz
#     ax.pcolormesh(
#         x,
#         y,
#         matriz,
#         shading="nearest",
#         vmin=matriz.min(),
#         vmax=matriz.max(),
#         cmap=cmap,
#         # edgecolors="gray",  # bordes negros
#         # linewidth=0.52,  # grosor del borde
#     )

#     # Configuración de electrodos
#     electrode_width = 0.2
#     electrode_height = 11

#     # Electrodo izquierdo
#     left_electrode = patches.Rectangle((-0.2, -0.5), electrode_width, electrode_height, color="gray", zorder=3)

#     # Electrodo derecho
#     right_electrode = patches.Rectangle((10.0, -0.5), electrode_width, electrode_height, color="gray", zorder=3)

#     ax.add_patch(left_electrode)
#     ax.add_patch(right_electrode)

#     # Configurar etiquetas y título
#     ax.set_xticks(np.arange(0, 11, 2))  # 🔹 Ticks cada 2 nm en X
#     ax.set_yticks(np.arange(0, 11, 2))  # 🔹 Ticks cada 2 nm en Y
#     ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
#     ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")
#     ax.set_title(rf"V_RRAM = {voltaje} V", pad=20)

#     # Ajustar formato visual
#     ax.set_aspect("equal")
#     ax.invert_yaxis()

#     # Ajustar límites del eje X y Y para que los electrodos sean visibles
#     ax.set_xlim(-0.15, 10.15)
#     ax.set_ylim(-0, 10)  # 🔥 Extiende el gráfico en Y para acomodar los electrodos

#     # Aumentar margen superior para más espacio en el título
#     plt.subplots_adjust(top=0.85)

#     # Guardar si se especifica un archivo
#     if filename and guardar_png:
#         plt.savefig(filename, bbox_inches="tight")

#     cadena = filename
#     ruta_pdf = os.path.splitext(cadena)[0] + ".pdf"
#     plt.savefig(ruta_pdf, bbox_inches="tight")
#     # ruta_svg = os.path.splitext(cadena)[0] + ".svg"
#     # plt.savefig(ruta_svg, bbox_inches="tight")

#     # Mostrar gráfico
#     plt.close(fig)

#     return None


def RepresentateState(
    matriz: np.ndarray,
    voltaje: float,
    filename: str | None = None,
    color=(0.0000, 0.0000, 0.5451),
    guardar_png: bool = False,
    device_size: float = 10e-9,
) -> None:
    """
    Representa el estado de una matriz de RRAM.
    Optimizado para publicación: electrodos ajustables, cero absoluto en la base,
    y colores estables independientemente de la ocupación de la matriz.
    """

    # El blanco SIEMPRE será el valor 0, y tu color SIEMPRE será el valor 1.
    cmap = ListedColormap([(1, 1, 1), color])
    fig, ax = plt.subplots(figsize=(12, 9))

    # Descomenta estas líneas según tu entorno
    # setup_paper_plt(plt, latex=True, scaling=3)
    # config_ax_state(ax)

    size_nm = device_size * 1e9  # Convertir a nm ya que siempre se representa en nm

    # 1. MATRIZ
    ax.imshow(
        matriz,
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[0, size_nm, 0, size_nm],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )

    # 2. ELECTRODOS
    electrode_width = 0.2
    y_start = 0
    electrode_height = size_nm

    left_electrode = patches.Rectangle(
        (-electrode_width, y_start), electrode_width, electrode_height, color="gray", zorder=1
    )

    right_electrode = patches.Rectangle((size_nm, y_start), electrode_width, electrode_height, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # Ajustar formato visual
    ax.set_aspect("equal")

    # 3. LÍMITES AJUSTADOS
    ax.set_xlim(-electrode_width, size_nm + electrode_width)

    # MEJORA 2: Un margen vertical invisible (0.05) para que el marco negro del gráfico
    # no "pise" ni ampute los píxeles de las vacantes en los extremos.
    margen_y = 0.05
    ax.set_ylim(-margen_y, size_nm + margen_y)

    # 4. MARCAS AUTOMÁTICAS
    paso_ticks = 2 if size_nm <= 15 else (5 if size_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, size_nm + 1, paso_ticks))
    ax.set_yticks(np.arange(0, size_nm + 1, paso_ticks))

    # Configurar etiquetas y título
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    ax.set_title(
        rf"$V_{{RRAM}}$ = {voltaje} V",
        pad=20,
    )

    plt.subplots_adjust(top=0.85)

    # Guardar archivos
    if filename:
        if guardar_png:
            plt.savefig(filename, bbox_inches="tight", dpi=150)

        # ruta_pdf = os.path.splitext(filename)[0] + ".pdf"
        # plt.savefig(ruta_pdf, bbox_inches="tight")

    plt.close(fig)

    return None


def RepresentateTwoStates(
    matriz1: np.ndarray,
    matriz2: np.ndarray,
    voltage: float,
    filename: str = None,  # type: ignore
    guardar_png: bool = False,
    device_size: float = 10e-9,
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

    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))  # Tamaño de la figura ajustado
    config_ax_state(ax)

    size_nm = device_size * 1e9  # Convertir a nm ya que siempre se representa en nm

    nrows, ncols = matriz1.shape
    x = np.linspace(0, size_nm, ncols)  # Escala real de 10 nm en eje X
    y = np.linspace(0, size_nm, nrows)  # Escala real de 10 nm en eje Y

    # Crear mapas de colores para cada matriz
    cmap1 = LinearSegmentedColormap.from_list("cmap1", [(1, 1, 1), (0.9647, 0.1725, 0.3059)], N=2)  # Rojo
    cmap2 = LinearSegmentedColormap.from_list("cmap2", [(1, 1, 1), (0.2314, 0.2275, 0.9647)], N=2)  # Azul

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
    electrode_height = size_nm  # Se extienden en Y (de -1 a 11)
    electrode_color = "gray"  # Color de los electrodos

    left_electrode = patches.Rectangle((-0.3, -0.5), electrode_width, electrode_height + 1, color=electrode_color)

    right_electrode = patches.Rectangle(
        (device_size, -0.5), electrode_width, electrode_height + 1, color=electrode_color
    )

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)
    # Aplicar configuración de ejes

    # Configurar etiquetas y título
    ax.set_xticks(np.arange(0, 26, 2))  # Ticks cada 2 nm en X
    ax.set_yticks(np.arange(0, 26, 2))  # Ticks cada 2 nm en Y
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")
    ax.set_title(rf"V = {voltage} (V)", pad=20)

    # Ajustar formato visual
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Ajustar límites del eje X y Y para que los electrodos sean visibles
    ax.set_xlim(-0.3, device_size + 0.3)
    ax.set_ylim(-0, device_size)

    # Aumentar margen superior para más espacio en el título
    plt.subplots_adjust(top=0.85)

    # Guardar si se especifica un archivo
    if filename and guardar_png:
        plt.savefig(filename, bbox_inches="tight")

    cadena = filename
    # ruta_pdf = os.path.splitext(cadena)[0] + ".pdf"
    # plt.savefig(ruta_pdf, bbox_inches="tight")

    ruta_pdf = os.path.splitext(cadena)[0] + ".png"
    plt.savefig(ruta_pdf, bbox_inches="tight", dpi=150)

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
    figures_path="C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/",
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
    setup_paper_plt(plt, latex=True, scaling=3)

    fig, axes = plt.subplots(figsize=(12, 9))
    config_ax_IV(axes)

    axes.set_xlabel("Voltage (V)")
    axes.set_ylabel("Current (A)")
    axes.set_yscale("log")
    axes.set_title(titulo_figura, pad=20)

    # ---------- EJE X ----------
    # Marcas exactas: -1.5 -1.0 −0.5 0.0 0.5 1.0
    axes.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    axes.set_xticklabels(["-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0"])

    # ---------- EJE Y ----------
    # Marcas en potencias de 10 de 10⁻⁷ a 10⁻²
    y_ticks = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    axes.set_yticks(y_ticks)
    axes.get_yaxis().set_major_formatter(ticker.FormatStrFormatter("$10^{%d}$"))

    # Esto imprime las etiquetas en forma 10⁻⁷, 10⁻⁶, etc.
    axes.set_yticklabels(
        [
            r"$10^{-9}$",
            r"$10^{-8}$",
            r"$10^{-7}$",
            r"$10^{-6}$",
            r"$10^{-5}$",
            r"$10^{-4}$",
            r"$10^{-3}$",
            r"$10^{-2}$",
        ]
    )

    # # Scatter de SET y RESET
    # axes.scatter(
    #     v_set,
    #     i_set,
    #     color="red",
    #     s=15,
    #     marker=markers.MarkerStyle("o"),
    #     facecolors="white",
    #     label="SET",
    # )
    # axes.scatter(
    #     v_reset,
    #     i_reset,
    #     color="red",
    #     s=15,
    #     marker=markers.MarkerStyle("s"),
    #     facecolors="white",
    #     label="RESET",
    # )

    # Represento una línea para el set y otra para el reset
    axes.plot(v_set, i_set, color="red", linewidth=4, label="SET")
    axes.plot(v_reset, i_reset, color="red", linewidth=4, label="RESET")

    # Ruta de los datos experimentales
    # ruta_archivo_set = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Mean_DC_Set_1t'
    # ruta_archivo_reset = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Mean_DC_Reset_1.txt'

    ruta_archivo_set = os.getcwd() + "Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt"
    ruta_archivo_reset = os.getcwd() + "Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt"

    # ruta_archivo_set = os.getcwd() + "/Datos_Experimentales/Medidas_Eduardo/D_Set_1_Run35.txt"
    # ruta_archivo_reset = os.getcwd() + "/Datos_Experimentales/Medidas_Eduardo/D_Reset_1_Run35.txt"

    # Cargar datos experimentales
    data_set = np.loadtxt(ruta_archivo_set, skiprows=1)
    data_reset = np.loadtxt(ruta_archivo_reset, skiprows=1)

    x_set = data_set[:, 2]
    y_set = data_set[:, 1]
    x_reset = data_reset[:, 2] * (-1.0)  # TODO: IMportante comprobar si las medidas se leen con el signo ya o no
    y_reset = abs(data_reset[:, 1])

    # Curvas experimentales
    axes.plot(x_set, y_set, "black", label="Set experimental", linewidth=2)
    axes.plot(x_reset, y_reset, "black", label="Reset experimental", linewidth=2)
    # Antes ponia 2.5 de grosor de linea (antes de las medidas de arturo)
    # Leyenda ajustada en la parte inferior izquierda
    axes.legend(
        labelspacing=0.3,
        handletextpad=0.2,
        handlelength=1.0,
        borderaxespad=0.2,
        loc="lower left",
    )

    # Guardar figura
    fig.savefig(figures_path + f"/I-V_{num_simulation + 1}.png", bbox_inches="tight", dpi=150)

    # # Guardar figura
    # fig.savefig(figures_path + f"/I-V_{num_simulation + 1}.pdf", bbox_inches="tight", dpi=150)
    # plt.close(fig)  # Cierra para liberar memoria

    # Guardar figura
    # fig.savefig(figures_path + f"/I-V_{num_simulation + 1}.svg", bbox_inches="tight", dpi=150)
    # plt.close(fig)  # Cierra para liberar memoria


def plot_IV_marcado(
    v_set,
    i_set,
    v_reset,
    i_reset,
    num_simulation,
    lista_puntos,
    desplazamiento,
    titulo_figura="I-V Characteristics",
    figures_path="C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/",
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
    setup_paper_plt(plt, latex=True, scaling=3)

    # plt.rcParams["axes.labelsize"] = 32

    fig, axes = plt.subplots(figsize=(12, 9))
    config_ax_IV(axes)

    axes.set_xlabel("Voltage (V)")
    axes.set_ylabel("Current (A)")
    axes.set_yscale("log")
    axes.set_title(titulo_figura, pad=20)

    # ---------- EJE X ----------
    # Marcas exactas: -1.5 -1.0 −0.5 0.0 0.5 1.0
    axes.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    axes.set_xticklabels(["-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0"])

    # ---------- EJE Y ----------
    # Marcas en potencias de 10 de 10⁻⁷ a 10⁻²
    y_ticks = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    axes.set_yticks(y_ticks)
    axes.get_yaxis().set_major_formatter(ticker.FormatStrFormatter("$10^{%d}$"))

    # Esto imprime las etiquetas en forma 10⁻⁷, 10⁻⁶, etc.
    axes.set_yticklabels(
        [
            r"$10^{-9}$",
            r"$10^{-8}$",
            r"$10^{-7}$",
            r"$10^{-6}$",
            r"$10^{-5}$",
            r"$10^{-4}$",
            r"$10^{-3}$",
            r"$10^{-2}$",
        ]
    )

    # # Scatter de SET y RESET
    # axes.scatter(
    #     v_set,
    #     i_set,
    #     color="red",
    #     s=15,
    #     marker=markers.MarkerStyle("o"),
    #     facecolors="white",
    #     label="SET",
    # )
    # axes.scatter(
    #     v_reset,
    #     i_reset,
    #     color="red",
    #     s=15,
    #     marker=markers.MarkerStyle("s"),
    #     facecolors="white",
    #     label="RESET",
    # )

    # Represento una línea para el set y otra para el reset
    axes.plot(v_set, i_set, color="red", linewidth=4, label="SET")
    axes.plot(v_reset, i_reset, color="red", linewidth=4, label="RESET")

    # Ruta de los datos experimentales
    # ruta_archivo_set = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Mean_DC_Set_1t'
    # ruta_archivo_reset = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Mean_DC_Reset_1.txt'
    # TODO: Seria ideal que estas rutas se pasaran como parámetros a la función para no tener que modificar el código cada vez que quiera cambiar los datos experimentales a mostrar, pero por ahora lo dejo así para avanzar con el resto de cosas
    # ruta_archivo_set = os.getcwd() + "/Datos_Experimentales/Medidas_Eduardo/D_Set_1_Run35.txt"
    # ruta_archivo_reset = os.getcwd() + "/Datos_Experimentales/Medidas_Eduardo/D_Reset_1_Run35.txt"

    ruta_archivo_set = os.getcwd() + "Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt"
    ruta_archivo_reset = os.getcwd() + "Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt"

    # Cargar datos experimentales
    data_set = np.loadtxt(ruta_archivo_set, skiprows=1)
    data_reset = np.loadtxt(ruta_archivo_reset, skiprows=1)

    x_set = data_set[:, 2]
    y_set = data_set[:, 1]
    x_reset = data_reset[:, 2] * (-1.0)
    y_reset = abs(data_reset[:, 1])

    (x_0, y_0) = next(iter(lista_puntos.values()))
    # print("Punto de referencia (0,0): ", (x_0, y_0))
    axes.scatter(
        1e-9,
        1e-9,
        color="blue",
        s=80,
        marker=markers.MarkerStyle("D"),
        zorder=10,
    )

    # Curvas experimentales
    axes.plot(x_set, y_set, "black", label="Set Exp.", linewidth=2)
    axes.plot(x_reset, y_reset, "black", label="Reset Exp.", linewidth=2)

    for label, (xp, yp) in lista_puntos.items():
        dx, factor_y = desplazamiento.get(label, (0.02, 1.0))  # 1.0 = sin desplazamiento en y
        # print(
        #     label,
        #     "puntos: ",
        #     (xp, yp),
        #     (dx, factor_y),
        #     "puntos finales: ",
        #     (xp + dx, yp * factor_y),
        # )
        print("Marcando punto: ", label, " en (", xp, ",", yp, ")")
        axes.scatter(xp, yp, color="blue", s=80, marker=markers.MarkerStyle("D"), zorder=10)
        axes.text(
            xp + dx,  # Usar la posición calculada en x
            max(yp * factor_y, 1e-6),  # Usar la posición calculada en y con un mínimo
            label,
            fontsize=42,
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
        dpi=150,
    )

    # # Guardar figura
    # fig.savefig(
    #     figures_path + f"/I-V_marcado_{num_simulation + 1}.pdf",
    #     bbox_inches="tight",
    #     dpi=150,
    # )
    # plt.close(fig)  # Cierra para liberar memoria

    # Guardar figura
    # fig.savefig(
    #     figures_path + f"/I-V_marcado_{num_simulation + 1}.svg",
    #     bbox_inches="tight",
    #     dpi=150,
    # )
    plt.close(fig)  # Cierra para liberar memoria


def plot_thermal_state(T_map, types_map, voltage, num_levels=10, device_size: float = 10e-9, save_path=None):
    """
    Visualiza el mapa de temperatura con superposición de materiales e isotermas alineadas.

    Argumentos:
    - T_map: Matriz de temperaturas (resultado del solver).
    - types_map: Matriz de materiales (ID 1: Filamento, ID 3: Electrodo).
    - title: Título del gráfico.
    - num_levels: Cantidad de líneas de contorno para las isotermas.
    - save_path: (Opcional) Ruta completa con nombre de archivo para guardar la imagen (ej: 'img/termico.png').
    """

    # 2. Aplicar los estilos globales ANTES de crear la figura
    setup_paper_plt(plt, latex=True, scaling=3)

    # 3. Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(12, 12))

    # 4. Aplicar configuración de estilo específica para los ejes
    config_ax_state(ax)

    # 1. Configuración de dimensiones y límites (Alineación perfecta)
    Ny, Nx = T_map.shape
    size_nm = device_size * 1e9  # Convertir a nm
    extent = [0, size_nm, 0, size_nm]

    # 5. Capa base: Temperatura
    im = ax.imshow(T_map, cmap="coolwarm", origin="lower", extent=extent, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Temperatura (K)")

    # 6. Capa de materiales (Overlay)
    overlay = np.zeros((Ny, Nx, 4))
    overlay[types_map == 3] = [0.2, 0.2, 0.2, 0.8]  # Electrodos: Gris oscuro
    overlay[types_map == 1] = [0.5, 0.5, 0.5, 0.4]  # Filamento: Gris claro
    ax.imshow(overlay, origin="lower", extent=extent, aspect="equal")

    # 7. Capa de Isotermas
    niveles = np.linspace(np.min(T_map), np.max(T_map), num_levels)
    contours = ax.contour(
        T_map,
        levels=niveles,
        colors="white",
        linewidths=1.5,
        alpha=1,
        origin="lower",
        extent=extent,
    )
    # Eliminado el fontsize=7 para que herede la proporción del documento
    ax.clabel(contours, fontsize=18, inline=True, fmt="%d")

    paso_ticks = 2 if size_nm <= 15 else (5 if size_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, size_nm + 1, paso_ticks))
    ax.set_yticks(np.arange(0, size_nm + 1, paso_ticks))

    # 8. Estética y Leyenda
    ax.set_title(f"$V_{{RRAM}}$ = {voltage} V", pad=25)  # Eliminado fontsize=14
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    # Ajustar formato visual
    ax.set_aspect("equal")
    # ax.invert_yaxis()

    # patches = [
    #     mpatches.Patch(color=(0.2, 0.2, 0.2, 0.8), label="Electrodos"),
    #     mpatches.Patch(color=(0.5, 0.5, 0.5, 0.4), label="Filamento"),
    #     Line2D([0], [0], color="black", lw=0.75, alpha=0.7, label="Isotermas"),
    # ]
    # ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.4, 1))

    plt.tight_layout()

    # 9. Guardar si se especifica la ruta
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # bbox_inches='tight' es crucial para que no recorte la leyenda exterior
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close(fig)  # Cierra para liberar memoria


# def plot_heatmap(data_map, title="Mapa de Distribución", cbar_label="Valor", cmap="viridis", save_path=None):
#     """
#     Visualiza un mapa de calor simple. Ideal para ver probabilidades (SET/RESET),
#     distribución de campos eléctricos, densidad de corriente, etc.

#     Argumentos:
#     - data_map: Matriz 2D de datos a representar (ej: mapa de probabilidades).
#     - title: Título del gráfico.
#     - cbar_label: Etiqueta de la barra de color (ej: 'Probabilidad', 'Campo (V/m)').
#     - cmap: Mapa de color de matplotlib (recomendados: 'viridis', 'plasma', 'magma', 'coolwarm').
#     - save_path: (Opcional) Ruta completa para guardar la imagen.
#     """

#     x = np.linspace(0, 25, 100)  # Escala real de 10 nm en eje X
#     y = np.linspace(0, 25, 100)  # Escala real de 10 nm en eje Y

#     # 2. Aplicar los estilos globales ANTES de crear la figura
#     setup_paper_plt(plt, latex=True, scaling=2.5)

#     # 3. Crear la figura y los ejes
#     fig, ax = plt.subplots(figsize=(12, 12))

#     # 3. Aplicar configuración de estilo específica para los ejes
#     config_ax_state(ax)

#     # 4. Representación de la matriz usando el objeto 'ax'
#     # Usamos origin='lower' para mantener la consistencia física donde y=0 es la base
#     im = ax.imshow(data_map, cmap=cmap, origin="lower", aspect="equal")

#     # 5. Barra de color asociada al objeto 'im' y al eje 'ax'
#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label(cbar_label)  # El tamaño de fuente ahora lo controla setup_paper_plt

#     # 6. Estética y etiquetas
#     ax.set_title(title, pad=15)
#     ax.set_xlabel("Ancho del Dispositivo (X)")
#     ax.set_ylabel("Grosor del Óxido (Y)")

#     plt.tight_layout()

#     # 7. Guardar si se especifica la ruta
#     if save_path:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")

#     plt.close(fig)  # Cierra para liberar memoria


def RepresentateHeatmap(
    matriz: np.ndarray,
    voltaje: float,
    titulo: str = "Heatmap",
    filename: str | None = None,
    cmap_name: str = "hot",
    label_colorbar: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    guardar_png: bool = False,
    electrode_width: float = 0.2,
    cero_blanco: bool = True,
    device_size: float = 10e-09,
) -> None:
    """
    Representa una matriz de valores continuos.
    Permite forzar que el valor 0 exacto se represente en color blanco.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Descomenta estas líneas según tu entorno si usas LaTeX
    setup_paper_plt(plt, latex=True, scaling=3)
    config_ax_state(ax)

    size_nm = device_size * 1e9  # Convertir a nm

    # 1. GESTIÓN DEL COLOR Y MÁSCARA PARA EL CERO
    # Obtenemos una copia del mapa de colores para poder modificarlo de forma segura
    cmap = plt.get_cmap(cmap_name).copy()

    if cero_blanco:
        # Le decimos al mapa de colores que los valores "inválidos" o enmascarados sean blancos
        cmap.set_bad(color="white")
        # Enmascaramos (ocultamos) todos los valores de la matriz que sean exactamente 0
        matriz_dibujo = np.ma.masked_where(matriz == 0, matriz)
    else:
        matriz_dibujo = matriz

    # Si no se pasan vmin/vmax, se calculan automáticamente
    val_min = vmin if vmin is not None else matriz.min()
    val_max = vmax if vmax is not None else matriz.max()

    # 2. MATRIZ
    im = ax.imshow(
        matriz_dibujo,  # <-- Usamos la matriz enmascarada
        cmap=cmap,
        vmin=val_min,
        vmax=val_max,
        extent=[0, size_nm, 0, size_nm],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )

    # 3. ELECTRODOS
    y_start = 0
    electrode_height = size_nm

    left_electrode = patches.Rectangle(
        (-electrode_width, y_start), electrode_width, electrode_height, color="gray", zorder=1
    )

    right_electrode = patches.Rectangle((size_nm, y_start), electrode_width, electrode_height, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # Ajustar formato visual
    ax.set_aspect("equal")

    # 4. LÍMITES AJUSTADOS
    ax.set_xlim(-electrode_width, size_nm + electrode_width)
    margen_y = 0.05
    ax.set_ylim(-margen_y, size_nm + margen_y)

    # Configurar marcas (ticks)
    paso_ticks = 2 if size_nm <= 15 else (5 if size_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, size_nm + 1, paso_ticks))
    ax.set_yticks(np.arange(0, size_nm + 1, paso_ticks))

    # Configurar etiquetas y título
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    ax.set_title(
        rf"{titulo} ($V_{{RRAM}}$ = {voltaje} V)",
        pad=20,
    )

    # 5. COLORBAR
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label_colorbar:
        cbar.set_label(label_colorbar)

    plt.subplots_adjust(top=0.85)

    # Guardar archivos
    if filename:
        if guardar_png:
            plt.savefig(filename, bbox_inches="tight", dpi=150)

        # ruta_pdf = os.path.splitext(filename)[0] + ".pdf"
        # plt.savefig(ruta_pdf, bbox_inches="tight")

    plt.close(fig)

    return None
