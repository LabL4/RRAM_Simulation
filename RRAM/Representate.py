import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib import markers

from matplotlib.colors import LinearSegmentedColormap

# import pandas as pd
import numpy as np
import os


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
    guardar_png: bool = True,
    device_size_x: float = 10e-9,
    device_size_y: float = 10e-9,
) -> None:
    """
    Representa el estado de una matriz de RRAM.
    Optimizado para publicación: electrodos ajustables, cero absoluto en la base,
    y colores estables independientemente de la ocupación de la matriz.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    La matriz se transpone antes de imshow para que X quede horizontal.
    """

    # El blanco SIEMPRE será el valor 0, y tu color SIEMPRE será el valor 1.
    cmap = ListedColormap([(1, 1, 1), color])
    fig, ax = plt.subplots(figsize=(12, 9))

    # Descomenta estas líneas según tu entorno
    setup_paper_plt(plt, latex=True, scaling=3)
    config_ax_state(ax)

    x_nm = device_size_x * 1e9  # Distancia entre electrodos en nm (eje horizontal)
    y_nm = device_size_y * 1e9  # Ancho transversal en nm (eje vertical)

    # 1. MATRIZ (transpuesta: shape[0]=X→horizontal, shape[1]=Y→vertical)
    ax.imshow(
        matriz.T,
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[0, x_nm, 0, y_nm],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )

    # 2. ELECTRODOS (barras verticales en X=0 y X=x_nm, altura = y_nm)
    electrode_width = 0.2
    y_start = 0
    electrode_height = y_nm

    left_electrode = patches.Rectangle(
        (-electrode_width, y_start), electrode_width, electrode_height, color="gray", zorder=1
    )

    right_electrode = patches.Rectangle((x_nm, y_start), electrode_width, electrode_height, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # 3. LÍMITES AJUSTADOS
    ax.set_xlim(-electrode_width, x_nm + electrode_width)

    margen_y = 0.05
    ax.set_ylim(-margen_y, y_nm + margen_y)

    # 4. MARCAS AUTOMÁTICAS
    paso_ticks_x = 2 if x_nm <= 15 else (5 if x_nm <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))

    # Configurar etiquetas y título
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    ax.set_title(
        rf"$V_{{RRAM}}$ = {voltaje} V",
        pad=20,
    )

    plt.subplots_adjust(top=0.85)

    # TODO Cambiar esto y que se guarde siempre
    # Guardar archivos
    if filename:
        if guardar_png:
            plt.savefig(filename, bbox_inches="tight", dpi=300)

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
    device_size_x: float = 10e-9,
    device_size_y: float = 10e-9,
) -> None:
    """
    Representa el estado de dos matrices con un estilo gráfico personalizado en el mismo plot.

    Parámetros:
    - matriz1 (np.ndarray): Primera matriz a representar (color rojo).
    - matriz2 (np.ndarray): Segunda matriz a representar (color azul).
    - voltage (float): Voltaje actual.
    - filename (str, opcional): Nombre del archivo para guardar la gráfica.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    """

    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))  # Tamaño de la figura ajustado
    config_ax_state(ax)

    x_nm = device_size_x * 1e9  # Distancia entre electrodos en nm (eje horizontal)
    y_nm = device_size_y * 1e9  # Ancho transversal en nm (eje vertical)

    # Crear mapas de colores para cada matriz
    cmap1 = LinearSegmentedColormap.from_list("cmap1", [(1, 1, 1), (0.9647, 0.1725, 0.3059)], N=2)  # Rojo
    cmap2 = LinearSegmentedColormap.from_list("cmap2", [(1, 1, 1), (0.2314, 0.2275, 0.9647)], N=2)  # Azul

    # Graficar la primera matriz (transpuesta: X→horizontal, Y→vertical)
    ax.imshow(
        matriz1.T,
        cmap=cmap1,
        vmin=matriz1.min(),
        vmax=matriz1.max(),
        extent=[0, x_nm, 0, y_nm],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )

    # Graficar la segunda matriz (transpuesta, con transparencia)
    ax.imshow(
        matriz2.T,
        cmap=cmap2,
        vmin=matriz2.min(),
        vmax=matriz2.max(),
        extent=[0, x_nm, 0, y_nm],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        alpha=0.5,
        zorder=3,
    )

    # Electrodos: barras verticales en X=0 y X=x_nm, altura = y_nm
    electrode_width = 0.2
    electrode_color = "gray"

    left_electrode = patches.Rectangle((-electrode_width, 0), electrode_width, y_nm, color=electrode_color)
    right_electrode = patches.Rectangle((x_nm, 0), electrode_width, y_nm, color=electrode_color)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # Configurar etiquetas y título
    paso_ticks_x = 2 if x_nm <= 15 else (5 if x_nm <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")
    ax.set_title(rf"V = {voltage} (V)", pad=20)

    # Límites ajustados
    ax.set_xlim(-electrode_width, x_nm + electrode_width)
    ax.set_ylim(-0.05, y_nm + 0.05)

    # Aumentar margen superior para más espacio en el título
    plt.subplots_adjust(top=0.85)

    # Guardar si se especifica un archivo
    if filename and guardar_png:
        plt.savefig(filename, bbox_inches="tight")

    cadena = filename
    # ruta_pdf = os.path.splitext(cadena)[0] + ".pdf"
    # plt.savefig(ruta_pdf, bbox_inches="tight")

    ruta_pdf = os.path.splitext(cadena)[0] + ".png"
    plt.savefig(ruta_pdf, bbox_inches="tight", dpi=300)

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

    ruta_archivo_set = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt"
    ruta_archivo_reset = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt"

    # ruta_archivo_set = os.getcwd() + "/Datos_Experimentales/Medidas_Eduardo/D_Set_1_Run35.txt"
    # ruta_archivo_reset = os.getcwd() + "/Datos_Experimentales/Medidas_Eduardo/D_Reset_1_Run35.txt"

    # Cargar datos experimentales
    data_set = np.loadtxt(ruta_archivo_set, skiprows=1)
    data_reset = np.loadtxt(ruta_archivo_reset, skiprows=1)

    x_set = data_set[:, 0]
    y_set = data_set[:, 1]
    x_reset = data_reset[:, 0]  # * (-1.0)   TODO: Importante comprobar si las medidas se leen con el signo ya o no
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
    fig.savefig(figures_path + f"/I-V_{num_simulation + 1}.png", bbox_inches="tight", dpi=300)

    # # Guardar figura
    # fig.savefig(figures_path + f"/I-V_{num_simulation + 1}.pdf", bbox_inches="tight", dpi=300)
    # plt.close(fig)  # Cierra para liberar memoria

    # Guardar figura
    # fig.savefig(figures_path + f"/I-V_{num_simulation + 1}.svg", bbox_inches="tight", dpi=300)
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

    ruta_archivo_set = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt"
    ruta_archivo_reset = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt"

    # Cargar datos experimentales
    data_set = np.loadtxt(ruta_archivo_set, skiprows=1)
    data_reset = np.loadtxt(ruta_archivo_reset, skiprows=1)

    x_set = data_set[:, 0]
    y_set = data_set[:, 1]
    x_reset = data_reset[:, 0]  # * (-1.0)   TODO: Importante comprobar si las medidas se leen con el signo ya o no
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
        print("\n-----------------------------")
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
        dpi=300,
    )

    # # Guardar figura
    # fig.savefig(
    #     figures_path + f"/I-V_marcado_{num_simulation + 1}.pdf",
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close(fig)  # Cierra para liberar memoria

    # Guardar figura
    # fig.savefig(
    #     figures_path + f"/I-V_marcado_{num_simulation + 1}.svg",
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    plt.close(fig)  # Cierra para liberar memoria


def plot_thermal_state(
    T_map,
    types_map,
    voltage,
    num_levels=10,
    device_size_x: float = 10e-9,
    device_size_y: float = 10e-9,
    atom_size: float = 0.25e-9,
    save_path=None,
):
    """
    Visualiza el mapa de temperatura con superposición de materiales e isotermas alineadas.

    Argumentos:
    - T_map: Matriz de temperaturas shape=(x_size+2, y_size), filas=X, cols=Y.
    - types_map: Matriz de materiales shape=(x_size+2, y_size).
    - num_levels: Cantidad de líneas de contorno para las isotermas.
    - save_path: (Opcional) Ruta completa con nombre de archivo para guardar la imagen.

    Convención de ejes:
        shape[0] = eje X (filas, incluyendo electrodos) → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    """

    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    # T_map tiene shape (x_size+2, y_size): se transpone para imshow
    x_size_ext, y_size = T_map.shape
    x_nm_ext = x_size_ext * atom_size * 1e9  # extensión X total incluyendo electrodos
    y_nm = device_size_y * 1e9
    extent = [0, x_nm_ext, 0, y_nm]

    # Capa base: Temperatura (transpuesta)
    im = ax.imshow(T_map.T, cmap="coolwarm", origin="lower", extent=extent, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Temperatura (K)")

    # Capa de materiales (Overlay, transpuesta)
    overlay = np.zeros((y_size, x_size_ext, 4))
    overlay[types_map.T == 3] = [0.2, 0.2, 0.2, 0.8]  # Electrodos: Gris oscuro
    overlay[types_map.T == 1] = [0.5, 0.5, 0.5, 0.4]  # Filamento: Gris claro
    ax.imshow(overlay, origin="lower", extent=extent, aspect="equal")

    # Capa de Isotermas (sobre la imagen transpuesta)
    niveles = np.linspace(np.min(T_map), np.max(T_map), num_levels)
    contours = ax.contour(
        T_map.T,
        levels=niveles,
        colors="white",
        linewidths=1.5,
        alpha=1,
        origin="lower",
        extent=extent,
    )
    ax.clabel(contours, fontsize=18, inline=True, fmt="%d")

    paso_ticks_x = 2 if x_nm_ext <= 15 else (5 if x_nm_ext <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm_ext + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))

    ax.set_title(f"$V_{{RRAM}}$ = {voltage} V", pad=25)
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


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
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")

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
    electrode_width: float = 0.2,
    cero_blanco: bool = True,
    device_size_x: float = 10e-09,
    device_size_y: float = 10e-09,
) -> None:
    """
    Representa una matriz de valores continuos.
    Permite forzar que el valor 0 exacto se represente en color blanco.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Descomenta estas líneas según tu entorno si usas LaTeX
    setup_paper_plt(plt, latex=True, scaling=3)
    config_ax_state(ax)

    x_nm = device_size_x * 1e9  # Distancia entre electrodos en nm (eje horizontal)
    y_nm = device_size_y * 1e9  # Ancho transversal en nm (eje vertical)

    # 1. GESTIÓN DEL COLOR Y MÁSCARA PARA EL CERO
    cmap = plt.get_cmap(cmap_name).copy()

    if cero_blanco:
        cmap.set_bad(color="white")
        matriz_dibujo = np.ma.masked_where(matriz == 0, matriz)
    else:
        matriz_dibujo = matriz

    # Si no se pasan vmin/vmax, se calculan automáticamente
    val_min = vmin if vmin is not None else matriz.min()
    val_max = vmax if vmax is not None else matriz.max()

    # 2. MATRIZ (transpuesta: X→horizontal, Y→vertical)
    im = ax.imshow(
        matriz_dibujo.T if not isinstance(matriz_dibujo, np.ma.MaskedArray) else np.ma.transpose(matriz_dibujo),
        cmap=cmap,
        vmin=val_min,
        vmax=val_max,
        extent=[0, x_nm, 0, y_nm],
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )

    # 3. ELECTRODOS (barras verticales en X=0 y X=x_nm)
    y_start = 0
    electrode_height = y_nm

    left_electrode = patches.Rectangle(
        (-electrode_width, y_start), electrode_width, electrode_height, color="gray", zorder=1
    )

    right_electrode = patches.Rectangle((x_nm, y_start), electrode_width, electrode_height, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # 4. LÍMITES AJUSTADOS
    ax.set_xlim(-electrode_width, x_nm + electrode_width)
    margen_y = 0.05
    ax.set_ylim(-margen_y, y_nm + margen_y)

    # Configurar marcas (ticks)
    paso_ticks_x = 2 if x_nm <= 15 else (5 if x_nm <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))

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
        plt.savefig(filename, bbox_inches="tight", dpi=300)

        # ruta_pdf = os.path.splitext(filename)[0] + ".pdf"
        # plt.savefig(ruta_pdf, bbox_inches="tight")

    plt.close(fig)

    return None


def plot_centros_filamento(
    matriz_state: np.ndarray,
    rangos_CF: list,
    centros_calculados: list,
    filas_intermedias: list,
    filename: str | None = None,
    device_size_x: float = 10e-9,
    device_size_y: float = 10e-9,
    atom_size: float = 0.25e-9,
    electrode_width: float = 0.2,
) -> None:
    """
    Genera un gráfico visualizando los centros de filamentos calculados y las
    columnas Y donde se colocarán los muros térmicos, envuelto por electrodos.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    Los centros y muros son posiciones en el eje Y (columnas) → líneas verticales en el plot.
    """
    # 1. Aplicar estilos y crear figura (Estándar RRAM)
    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    x_nm = device_size_x * 1e9
    y_nm = device_size_y * 1e9
    atom_size_nm = atom_size * 1e9

    # 2. Dibujo de la matriz transpuesta (X→horizontal, Y→vertical)
    ax.imshow(matriz_state.T, cmap="Blues", origin="lower", aspect="equal", extent=[0, x_nm, 0, y_nm], zorder=2)

    # ==========================================================
    # 3. ELECTRODOS ENVOLVENTES (barras verticales en X=0 y X=x_nm)
    # ==========================================================
    left_electrode = patches.Rectangle((-electrode_width, 0), electrode_width, y_nm, color="gray", zorder=1)
    right_electrode = patches.Rectangle((x_nm, 0), electrode_width, y_nm, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # ==========================================================
    # 4. DIBUJO DE LÍNEAS CON LEYENDAS DINÁMICAS
    # Los rangos, centros y muros son posiciones en el eje Y (columnas).
    # Tras la transposición, el eje Y del array se mapea al eje vertical del plot.
    # ==========================================================
    # Límites de los rangos (líneas horizontales en el eje Y del plot)
    for idx, (col_min, col_max) in enumerate(rangos_CF):
        if idx < len(rangos_CF) - 1:
            ax.axhline(
                y=(col_max + 0.5) * atom_size_nm,
                color="gray",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label="Límite de Rango" if idx == 0 else "",
                zorder=3,
            )

    # CENTROS calculados (posiciones Y → líneas horizontales en el plot)
    for idx, centro in enumerate(centros_calculados):
        if centro is not None:
            ax.axhline(
                y=(centro + 0.5) * atom_size_nm,
                color="#D32F2F",
                linestyle="-",
                alpha=0.9,
                linewidth=3,
                label=f"Centro {idx + 1} (Col {centro})",
                zorder=3,
            )

    # Muros térmicos (posiciones Y → líneas horizontales en el plot)
    for idx, col_mid in enumerate(filas_intermedias):
        if col_mid is not None:
            ax.axhline(
                y=(col_mid + 0.5) * atom_size_nm,
                color="#388E3C",
                linestyle="-.",
                alpha=0.9,
                linewidth=2.5,
                label=f"Muro {idx + 1} (Col {col_mid})",
                zorder=3,
            )

    # ==========================================================
    # 5. LÍMITES AJUSTADOS Y ESTÉTICA
    # ==========================================================
    ax.set_xlim(-electrode_width, x_nm + electrode_width)
    margen_y = 0.05
    ax.set_ylim(-margen_y, y_nm + margen_y)

    paso_ticks_x = 2 if x_nm <= 15 else (5 if x_nm <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))

    ax.set_title("Centros de Filamento y Límites", pad=20)
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    # <-- ¡Mejora en la Leyenda! La colocamos fuera de la gráfica a la derecha
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.95)

    plt.subplots_adjust(top=0.85)

    # ==========================================================
    # 6. GUARDADO Y CIERRE
    # ==========================================================
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Es vital bbox_inches="tight" para que la leyenda exterior no se recorte al guardar
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close(fig)
    return None


def plot_centros_filamento_det(
    matriz_state: np.ndarray,
    rangos_CF: list,
    centros_calculados: list,
    filas_intermedias: list,
    filename: str | None = None,
    device_size_x: float = 10e-9,
    device_size_y: float = 10e-9,
    atom_size: float = 0.25e-9,
    electrode_width: float = 0.2,
) -> None:
    """
    Genera un gráfico visualizando los centros de filamentos calculados y las
    columnas Y donde se colocarán los muros térmicos, envuelto por electrodos.
    Incluye una malla visual estricta cada atom_size nm.

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    Los centros y muros son posiciones en el eje Y (columnas) → líneas horizontales en el plot.
    """
    # 1. Aplicar estilos y crear figura (Estándar RRAM)
    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    x_nm = device_size_x * 1e9
    y_nm = device_size_y * 1e9
    atom_size_nm = atom_size * 1e9

    # 2. Dibujo de la matriz transpuesta (X→horizontal, Y→vertical)
    ax.imshow(matriz_state.T, cmap="Blues", origin="lower", aspect="equal", extent=[0, x_nm, 0, y_nm], zorder=2)

    # ==========================================================
    # 3. ELECTRODOS ENVOLVENTES (barras verticales en X=0 y X=x_nm)
    # ==========================================================
    left_electrode = patches.Rectangle((-electrode_width, 0), electrode_width, y_nm, color="gray", zorder=1)
    right_electrode = patches.Rectangle((x_nm, 0), electrode_width, y_nm, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # ==========================================================
    # 3.5. CUADRÍCULA ESTRICTA CADA atom_size NM (eje Y del plot = columnas Y del array)
    # ==========================================================
    for y_grid in np.arange(0, y_nm + atom_size_nm, atom_size_nm):
        ax.axhline(
            y=y_grid,
            color="black",
            linestyle="--",
            alpha=0.25,
            linewidth=0.8,
            zorder=2.5,
        )

    # ==========================================================
    # 4. DIBUJO DE LÍNEAS CON LEYENDAS DINÁMICAS
    # Los rangos, centros y muros son posiciones en el eje Y (columnas).
    # Tras la transposición, el eje Y del array se mapea al eje vertical del plot.
    # ==========================================================
    # Límites de los rangos (líneas horizontales en el eje Y del plot)
    for idx, (col_min, col_max) in enumerate(rangos_CF):
        if idx < len(rangos_CF) - 1:
            ax.axhline(
                y=(col_max + 0.5) * atom_size_nm,
                color="gray",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label="Límite de Rango" if idx == 0 else "",
                zorder=3,
            )

    # CENTROS calculados (posiciones Y → líneas horizontales en el plot)
    for idx, centro in enumerate(centros_calculados):
        if centro is not None:
            ax.axhline(
                y=(centro + 0.5) * atom_size_nm,
                color="#D32F2F",
                linestyle="-",
                alpha=0.9,
                linewidth=3,
                label=f"Centro {idx + 1} (Col {centro})",
                zorder=3,
            )

    # Muros térmicos (posiciones Y → líneas horizontales en el plot)
    for idx, col_mid in enumerate(filas_intermedias):
        if col_mid is not None:
            ax.axhline(
                y=(col_mid + 0.5) * atom_size_nm,
                color="#388E3C",
                linestyle="-.",
                alpha=0.9,
                linewidth=2.5,
                label=f"Muro {idx + 1} (Col {col_mid})",
                zorder=3,
            )

    # ==========================================================
    # 5. LÍMITES AJUSTADOS Y ESTÉTICA
    # ==========================================================
    ax.set_xlim(-electrode_width, x_nm + electrode_width)
    margen_y = 0.05
    ax.set_ylim(-margen_y, y_nm + margen_y)

    paso_ticks_x = 2 if x_nm <= 15 else (5 if x_nm <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))

    ax.set_title("Centros de Filamento y Límites", pad=20)
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    # Leyenda fuera de la gráfica a la derecha
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.95)

    plt.subplots_adjust(top=0.85)

    # ==========================================================
    # 6. GUARDADO Y CIERRE
    # ==========================================================
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Es vital bbox_inches="tight" para que la leyenda exterior no se recorte al guardar
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close(fig)
    return None


def plot_muro_termico(
    matriz_muros: np.ndarray,
    matriz_molde: np.ndarray | None = None,
    titulo: str = "Thermal Walls Placement and Temperature",
    filename: str | None = None,
    device_size_x: float = 10e-09,
    device_size_y: float = 10e-09,
    electrode_width: float = 0.2,
) -> None:
    """
    Dibuja el dispositivo en 2D. Muestra los filamentos de fondo (gris claro),
    la ubicación de los muros térmicos (colores) y los electrodos (gris oscuro).

    Convención de ejes:
        shape[0] = eje X (filas) = dirección entre electrodos → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    """
    # 1. Aplicar estilos estándar
    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    x_nm = device_size_x * 1e9
    y_nm = device_size_y * 1e9

    # ==========================================================
    # 2. CAPA DE FONDO: Matriz Molde (Filamentos) — transpuesta
    # ==========================================================
    if matriz_molde is not None:
        ax.imshow(
            matriz_molde.T,
            cmap="Greys",
            origin="lower",
            alpha=0.25,
            extent=[0, x_nm, 0, y_nm],
            aspect="equal",
            zorder=2,
        )

    # ==========================================================
    # 3. CAPA DEL MURO: Temperaturas Fijas — transpuesta, ceros enmascarados
    # ==========================================================
    muros_visibles = np.ma.masked_where(matriz_muros == 0, matriz_muros)

    im_muros = ax.imshow(
        np.ma.transpose(muros_visibles),
        cmap="plasma",
        origin="lower",
        extent=[0, x_nm, 0, y_nm],
        aspect="equal",
        zorder=3,
    )

    cbar = fig.colorbar(im_muros, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fixed Boundary Temperature (K)")

    # ==========================================================
    # 4. ELECTRODOS (barras verticales en X=0 y X=x_nm)
    # ==========================================================
    left_electrode = patches.Rectangle((-electrode_width, 0), electrode_width, y_nm, color="gray", zorder=1)
    right_electrode = patches.Rectangle((x_nm, 0), electrode_width, y_nm, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # ==========================================================
    # 5. CONFIGURACIÓN DE EJES Y ESTÉTICA
    # ==========================================================
    ax.set_xlim(-electrode_width, x_nm + electrode_width)
    margen_y = 0.05
    ax.set_ylim(-margen_y, y_nm + margen_y)

    paso_ticks_x = 2 if x_nm <= 15 else (5 if x_nm <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))

    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")
    ax.set_title(titulo, pad=20)

    plt.subplots_adjust(top=0.85)

    # ==========================================================
    # 6. GUARDADO DE ARCHIVO
    # ==========================================================
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    plt.close(fig)

    return None


def plot_thermal_state_muro(
    T_map,
    types_map,
    voltage,
    num_levels=10,
    device_size_x: float = 10e-9,
    device_size_y: float = 10e-9,
    atom_size: float = 0.25e-9,
    save_path=None,
    filas_intermedias: list | None = None,
):
    """
    Visualiza el mapa de temperatura con superposición de materiales e isotermas alineadas.
    Permite visualizar la ubicación de los muros térmicos si se proporcionan las columnas Y.

    Argumentos:
    - T_map: Matriz de temperaturas shape=(x_size+2, y_size), filas=X, cols=Y.
    - types_map: Matriz de materiales shape=(x_size+2, y_size).
    - filas_intermedias: Posiciones en el eje Y (columnas) donde se ubican los muros.

    Convención de ejes:
        shape[0] = eje X (filas, incluyendo electrodos) → eje horizontal del plot.
        shape[1] = eje Y (columnas) = ancho transversal → eje vertical del plot.
    """

    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    # T_map tiene shape (x_size+2, y_size)
    x_size_ext, y_size = T_map.shape
    x_nm_ext = x_size_ext * atom_size * 1e9  # extensión X total incluyendo electrodos
    y_nm = device_size_y * 1e9
    atom_size_nm = atom_size * 1e9
    extent = [0, x_nm_ext, 0, y_nm]

    # Capa base: Temperatura (transpuesta)
    im = ax.imshow(T_map.T, cmap="coolwarm", origin="lower", extent=extent, aspect="equal", zorder=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Temperatura (K)")

    # Capa de materiales (Overlay, transpuesta)
    overlay = np.zeros((y_size, x_size_ext, 4))
    overlay[types_map.T == 3] = [0.2, 0.2, 0.2, 0.8]  # Electrodos: Gris oscuro
    overlay[types_map.T == 1] = [0.5, 0.5, 0.5, 0.4]  # Filamento: Gris claro
    ax.imshow(overlay, origin="lower", extent=extent, aspect="equal", zorder=2)

    # Capa de Isotermas (sobre imagen transpuesta)
    niveles = np.linspace(np.min(T_map), np.max(T_map), num_levels)
    contours = ax.contour(
        T_map.T, levels=niveles, colors="white", linewidths=1.5, alpha=1, origin="lower", extent=extent, zorder=3
    )
    ax.clabel(contours, fontsize=18, inline=True, fmt="%d")

    # ==========================================================
    # MUROS TÉRMICOS: posiciones en el eje Y (columnas).
    # Tras transponer, el eje Y del array → eje vertical del plot → axhline.
    # ==========================================================
    if filas_intermedias is not None:
        for idx, col_mid in enumerate(filas_intermedias):
            if col_mid is not None:
                ax.axhline(
                    y=(col_mid + 0.5) * atom_size_nm,
                    color="#00FFFF",
                    linestyle="--",
                    alpha=0.9,
                    linewidth=3,
                    zorder=4,
                    label="Thermal Wall" if idx == 0 else "",
                )
                ax.axhline(
                    y=(col_mid + 1 + 0.5) * atom_size_nm,
                    color="#00FFFF",
                    linestyle="--",
                    alpha=0.9,
                    linewidth=3,
                    zorder=4,
                )
    # ==========================================================

    paso_ticks_x = 2 if x_nm_ext <= 15 else (5 if x_nm_ext <= 30 else 10)
    paso_ticks_y = 2 if y_nm <= 15 else (5 if y_nm <= 30 else 10)
    ax.set_xticks(np.arange(0, x_nm_ext + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, y_nm + 1, paso_ticks_y))

    ax.set_title(f"$V_{{RRAM}}$ = {voltage} V", pad=25)
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_perfil_temperatura(
    distancias: np.ndarray,
    perfiles: dict,
    step: int = 1,
    title: str = "Vertical Temperature Profile Evolution x = 20",
    save_path: str | None = None,
    usar_latex: bool = True,
    scaling: float = 3,
):
    """
    Dibuja múltiples perfiles de temperatura en una misma gráfica,
    manteniendo la estética exacta de línea y marcadores.
    """
    # 1. Aplicar los estilos globales ANTES de crear la figura
    setup_paper_plt(plt, latex=usar_latex, scaling=scaling)

    # 2. Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(16, 12))

    # 3. Aplicar la configuración de estilo específica para los ejes (Grid, ticks, etc.)
    config_ax(ax)

    # 4. Generar una paleta de colores dinámicos (para que se distingan las líneas)
    viridis = plt.get_cmap("viridis")  # Obtener el colormap 'viridis'
    colores = viridis(np.linspace(0, 0.9, len(perfiles)))

    # 5. Dibujar cada línea con la MISMA ESTÉTICA que tu función original
    for (etiqueta, data_y), color in zip(perfiles.items(), colores):
        ax.plot(
            distancias,
            data_y,
            color=color,  # El color cambia en cada iteración
            linewidth=2.5,
            label=etiqueta,  # La leyenda corresponde a cada voltaje o iteración
            marker="o",
            markersize=6 * (scaling / 2.5),
            markevery=step,
            markerfacecolor="white",  # Mantiene el interior del punto blanco
            markeredgewidth=1.5,  # Mantiene el grosor del borde
        )

    # 6. Configurar etiquetas y estética usando comandos de siunitx
    ax.set_xlabel(r"Vertical Distance (\si{\nano\meter})")
    ax.set_ylabel(r"Temperature (K)")
    ax.set_title(title, pad=15)

    # Añadir la leyenda para saber qué es cada línea
    ax.legend(loc="upper right")

    plt.tight_layout()

    # 7. Guardar la gráfica
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # Opcional: guardar en pdf también
        # ruta_pdf = os.path.splitext(save_path)[0] + ".pdf"
        # plt.savefig(ruta_pdf, bbox_inches="tight")

    plt.close(fig)

    return None
