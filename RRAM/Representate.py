import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib import markers

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

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


# setup_paper_plt(plt, latex=True, scaling=3)

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def guardar_figura(fig: Figure, ruta_base: str, extension: str, dpi: int = 300) -> None:
    """
    Guarda una figura de matplotlib en el formato especificado.

    Args:
        fig (plt.Figure): La figura de matplotlib que se desea guardar.
        ruta_base (str): Ruta completa y nombre del archivo de salida (sin la extensión).
        extension (str): Formato de guardado deseado (ej. 'png', 'pdf', 'svg').
        dpi (int, opcional): Resolución de la imagen. Por defecto es 300.
    """
    # Convertimos a minúsculas por seguridad (ej: "PNG" -> "png")
    formato = extension.lower()

    match formato:
        case "png":
            fig.savefig(f"{ruta_base}.png", bbox_inches="tight", dpi=dpi)
        case "pdf":
            fig.savefig(f"{ruta_base}.pdf", bbox_inches="tight", dpi=dpi)
        case "svg":
            fig.savefig(f"{ruta_base}.svg", bbox_inches="tight", dpi=dpi)
        case _:
            # Caso por defecto si se introduce una extensión no contemplada
            logger.info(f"Advertencia: Formato '{formato}' no soportado. Guardando como .png")
            fig.savefig(f"{ruta_base}.png", bbox_inches="tight", dpi=dpi)


# # (0.9647, 0.1725, 0.3059) color rojo original


def RepresentateState(
    matriz: np.ndarray,
    voltaje: float,
    filename: str,
    color=(0.0000, 0.0000, 0.5451),
    guardar_png: bool = False,
    atom_size: float = 0.25e-9,  # Tamaño físico de una celda en metros (ej: 0.2 nm)
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
    setup_paper_plt(plt, latex=True, scaling=3)
    config_ax_state(ax)

    # 1. CÁLCULO DE DIMENSIONES (Crecimiento en X, Clasificación en Y)
    device_size_y, device_size_x = matriz.shape  # ¡Fíjate en el orden Y, X!

    size_x_nm = device_size_x * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm
    size_y_nm = device_size_y * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm

    logger.info(f"Dimensiones del dispositivo: {size_x_nm:.2f} nm x {size_y_nm:.2f} nm")

    # 1. MATRIZ
    ax.imshow(
        matriz,
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=(0.0, size_x_nm, 0.0, size_y_nm),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )

    # 2. ELECTRODOS
    electrode_width = 0.2
    y_start = 0
    electrode_height = size_y_nm

    left_electrode = patches.Rectangle(
        (-electrode_width, y_start), electrode_width, electrode_height, color="gray", zorder=1
    )

    right_electrode = patches.Rectangle((size_x_nm, y_start), electrode_width, electrode_height, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # Ajustar formato visual
    ax.set_aspect("equal")
    ax.set_aspect("equal", adjustable="box")

    # 3. LÍMITES AJUSTADOS
    ax.set_xlim(-electrode_width, size_x_nm + electrode_width)

    # MEJORA 2: Un margen vertical invisible (0.05) para que el marco negro del gráfico
    # no "pise" ni ampute los píxeles de las vacantes en los extremos.
    margen_y = 0.05
    ax.set_ylim(-margen_y, size_y_nm + margen_y)

    # 4. MARCAS AUTOMÁTICAS
    paso_ticks_x = 2 if size_x_nm <= 15 else (5 if size_x_nm <= 30 else 10)
    paso_ticks_y = 2 if size_y_nm <= 15 else (5 if size_y_nm <= 30 else 10)

    ax.set_xticks(np.arange(0, size_x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, size_y_nm + 1, paso_ticks_y))

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
    if guardar_png:
        plt.savefig(filename, bbox_inches="tight", dpi=300)
    else:
        ruta_pdf = os.path.splitext(filename)[0] + ".pdf"
        plt.savefig(ruta_pdf, bbox_inches="tight")

    plt.close(fig)

    return None


def RepresentateTwoStates(
    matriz1: np.ndarray,
    matriz2: np.ndarray,
    filename: str,
    voltage: float,
    guardar_png: bool = False,
    atom_size: float = 0.25e-9,
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

    fig, ax = plt.subplots(figsize=(12, 9))  # Tamaño de la figura ajustado

    setup_paper_plt(plt, latex=True, scaling=3)
    config_ax_state(ax)

    # 1. CÁLCULO DE DIMENSIONES (Crecimiento en X, Clasificación en Y)
    device_size_x, device_size_y = matriz1.shape

    size_x_nm = device_size_x * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm
    size_y_nm = device_size_y * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm

    nrows, ncols = matriz1.shape
    x = np.linspace(0, size_x_nm, ncols)  # Escala real de 10 nm en eje X
    y = np.linspace(0, size_y_nm, nrows)  # Escala real de 10 nm en eje Y

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
    electrode_height = size_y_nm  # Se extienden en Y (de -1 a 11)
    electrode_color = "gray"  # Color de los electrodos

    left_electrode = patches.Rectangle((-0.3, -0.5), electrode_width, electrode_height + 1, color=electrode_color)

    right_electrode = patches.Rectangle((size_x_nm, -0.5), electrode_width, electrode_height + 1, color=electrode_color)

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
    ax.set_xlim(-0.3, size_x_nm + 0.3)
    ax.set_ylim(-0, size_y_nm)

    # Aumentar margen superior para más espacio en el título
    plt.subplots_adjust(top=0.85)

    # Guardar si se especifica un archivo
    if guardar_png:
        plt.savefig(filename, bbox_inches="tight")

        cadena = filename
        ruta_pdf = os.path.splitext(filename)[0] + ".pdf"
        plt.savefig(ruta_pdf, bbox_inches="tight")

        ruta_pdf = os.path.splitext(cadena)[0] + ".png"
        plt.savefig(filename, bbox_inches="tight", dpi=300)

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
    extension_guardado="png",
):
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
        extension_guardado (str): Format to save the figure ('png', 'pdf', 'svg').
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

    # Represento una línea para el set y otra para el reset
    axes.plot(v_set, i_set, color="red", linewidth=4, label="SET")
    axes.plot(v_reset, i_reset, color="red", linewidth=4, label="RESET")

    # Ruta de los datos experimentales
    # ruta_archivo_set = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Mean_DC_Set_1t'
    # ruta_archivo_reset = 'C:/Users/Usuario/Documents/GitHub/RRAM_Simulation/Datos_Experimentales/Ciclos_Experimentales/Mean_DC_Reset_1.txt'

    ruta_archivo_set = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1600.txt"
    ruta_archivo_reset = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1600.txt"

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

    # ---------- GUARDADO DE LA FIGURA ----------

    # Base de la ruta sin extensión para simplificar el código de guardado
    ruta_base = f"{figures_path}/I-V_{num_simulation + 1}"

    # Convertimos a minúsculas por seguridad (ej: "PNG" -> "png")
    match extension_guardado.lower():
        case "png":
            fig.savefig(f"{ruta_base}.png", bbox_inches="tight", dpi=300)
        case "pdf":
            fig.savefig(f"{ruta_base}.pdf", bbox_inches="tight", dpi=300)
        case "svg":
            fig.savefig(f"{ruta_base}.svg", bbox_inches="tight", dpi=300)
        case _:
            # Caso por defecto si se introduce una extensión no contemplada
            logger.info(f"Advertencia: Formato '{extension_guardado}' no soportado. Guardando como .png")
            fig.savefig(f"{ruta_base}.png", bbox_inches="tight", dpi=300)

    # Cierre de la figura unificado para todos los casos (evita fugas de memoria)
    plt.close(fig)

    return None


def plot_IV_marcado(
    v_set,
    i_set,
    v_reset,
    i_reset,
    num_simulation,
    lista_puntos,
    desplazamiento,
    extension_guardado="png",
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

    # Represento una línea para el set y otra para el reset
    axes.plot(v_set, i_set, color="red", linewidth=4, label="SET")
    axes.plot(v_reset, i_reset, color="red", linewidth=4, label="RESET")

    # Ruta de los datos experimentales
    ruta_archivo_set = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1600.txt"
    ruta_archivo_reset = os.getcwd() + "/Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1600.txt"

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

    # ---------- GUARDADO DE LA FIGURA ----------

    # Base de la ruta sin extensión para simplificar el código de guardado
    ruta_base = f"{figures_path}/I-V_marcado_{num_simulation + 1}"

    # Convertimos a minúsculas por seguridad (ej: "PNG" -> "png")
    match extension_guardado.lower():
        case "png":
            fig.savefig(f"{ruta_base}.png", bbox_inches="tight", dpi=300)
        case "pdf":
            fig.savefig(f"{ruta_base}.pdf", bbox_inches="tight", dpi=300)
        case "svg":
            fig.savefig(f"{ruta_base}.svg", bbox_inches="tight", dpi=300)
        case _:
            # Caso por defecto si se introduce una extensión no contemplada
            logger.info(f"Advertencia: Formato '{extension_guardado}' no soportado. Guardando como .png")
            fig.savefig(f"{ruta_base}.png", bbox_inches="tight", dpi=300)

    # Cierre de la figura unificado para todos los casos (evita fugas de memoria)
    plt.close(fig)


def plot_thermal_state(T_map, types_map, voltage, num_levels=10, atom_size: float = 0.25e-9, save_path=None):
    """
    Visualiza el mapa de temperatura con superposición de materiales e isotermas alineadas.

    Argumentos:
    - T_map: Dato de temperatura. Debe ser una matriz 2D (np.ndarray de ndim==2).
             Si es escalar o 1D se descarta y la función retorna sin generar figura.
    - types_map: Matriz de materiales (ID 1: Filamento, ID 3: Electrodo).
    - voltage: Voltaje aplicado (usado en el título).
    - num_levels: Cantidad de líneas de contorno para las isotermas.
    - save_path: (Opcional) Ruta completa con nombre de archivo para guardar la imagen.
    """

    # Guardia: solo procesar si T_map es una matriz 2D con variación térmica real
    if not isinstance(T_map, np.ndarray) or T_map.ndim < 2:
        logger.debug("plot_thermal_state: T_map no es matriz 2D — salto.")
        return
    if np.ptp(T_map) == 0:
        logger.debug("plot_thermal_state: T_map uniforme (sin gradiente) — salto isotermas.")
        # Continúa pero sin isotermas (ver abajo)

    # 2. Aplicar los estilos globales ANTES de crear la figura
    setup_paper_plt(plt, latex=True, scaling=3)

    # 3. Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(12, 9))

    # 4. Aplicar configuración de estilo específica para los ejes
    config_ax_state(ax)

    # 1. Configuración de dimensiones y límites (Alineación perfecta)
    Ny, Nx = T_map.shape
    size_x_nm = Nx * atom_size * 1e9  # Convertir a nm
    size_y_nm = Ny * atom_size * 1e9  # Convertir a nm

    extent = (0, size_x_nm, 0, size_y_nm)

    # 5. Capa base: Temperatura
    im = ax.imshow(T_map, cmap="coolwarm", origin="lower", extent=extent, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Temperatura (K)")

    # 6. Capa de materiales (Overlay)
    overlay = np.zeros((Ny, Nx, 4))
    overlay[types_map == 3] = [0.2, 0.2, 0.2, 0.8]  # Electrodos: Gris oscuro
    overlay[types_map == 1] = [0.5, 0.5, 0.5, 0.4]  # Filamento: Gris claro
    ax.imshow(overlay, origin="lower", extent=extent, aspect="equal")

    # 7. Capa de Isotermas (solo si hay gradiente real)
    if np.ptp(T_map) > 0:
        niveles = np.linspace(np.min(T_map), np.max(T_map), num_levels)
        # Eliminar duplicados que puedan surgir por precisión numérica
        niveles = np.unique(niveles)
        if len(niveles) > 1:
            contours = ax.contour(
                T_map,
                levels=niveles,
                colors="white",
                linewidths=1.5,
                alpha=1,
                origin="lower",
                extent=extent,
            )
            ax.clabel(contours, fontsize=18, inline=True, fmt="%d")

    paso_ticks_x = 2 if size_x_nm <= 15 else (5 if size_x_nm <= 30 else 10)
    paso_ticks_y = 2 if size_y_nm <= 15 else (5 if size_y_nm <= 30 else 10)

    ax.set_xticks(np.arange(0, size_x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, size_y_nm + 1, paso_ticks_y))

    # 8. Estética y Leyenda
    ax.set_title(f"$V_{{RRAM}}$ = {voltage} V", pad=25)  # Eliminado fontsize=14
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    # Ajustar formato visual
    ax.set_aspect("equal")

    plt.tight_layout()

    # 9. Guardar si se especifica la ruta
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # bbox_inches='tight' es crucial para que no recorte la leyenda exterior
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)  # Cierra para liberar memoria


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
        extent=(0, size_nm, 0, size_nm),
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
        plt.savefig(filename, bbox_inches="tight", dpi=300)

        # ruta_pdf = os.path.splitext(filename)[0] + ".pdf"
        # plt.savefig(ruta_pdf, bbox_inches="tight")

    plt.close(fig)

    return None


def plot_centros_filamento_det(
    matriz_state: np.ndarray,
    rangos_CF: list,
    centros_calculados: list,
    filas_intermedias: list,
    filename: str,
    atom_size: float = 0.25e-9,
) -> None:
    """
    Genera un gráfico visualizando los centros de filamentos calculados y las
    filas donde se colocarán los muros térmicos, envuelto por electrodos.
    Muestra en la leyenda el índice y la fila exacta de cada centro y muro.
    Incluye una malla visual estricta cada 0.25 nm (en los bordes de la celda).
    """
    # 1. Aplicar estilos y crear figura (Estándar RRAM)
    fig, ax = plt.subplots(figsize=(12, 9))

    setup_paper_plt(plt, latex=True, scaling=3)
    config_ax_state(ax)

    device_size_x, device_size_y = matriz_state.shape

    size_x_nm = device_size_x * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm
    size_y_nm = device_size_y * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm

    atom_size_nm = atom_size * 1e9

    # 2. Dibujo de la matriz (zorder=2)
    ax.imshow(matriz_state, cmap="Blues", origin="lower", aspect="equal", extent=(0, size_x_nm, 0, size_y_nm), zorder=2)

    # ==========================================================
    # 3. ELECTRODOS ENVOLVENTES
    # =========================================================
    electrode_width = 0.2
    y_start = 0
    electrode_height = size_y_nm

    left_electrode = patches.Rectangle(
        (-electrode_width, y_start), electrode_width, electrode_height, color="gray", zorder=1
    )
    right_electrode = patches.Rectangle((size_x_nm, y_start), electrode_width, electrode_height, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # ==========================================================
    # 3.5. CUADRÍCULA ESTRICTA CADA 0.25 NM
    # ==========================================================
    # Dibujamos líneas horizontales desde 0 hasta size_nm saltando exactamente el tamaño del átomo
    for y_grid in np.arange(0, size_x_nm + atom_size_nm, atom_size_nm):
        ax.axhline(
            y=y_grid,
            color="black",  # Color oscuro para contraste
            linestyle="--",  # Línea discontinua
            alpha=0.25,  # Transparencia baja para que no sature
            linewidth=0.8,  # Línea fina
            zorder=2.5,  # Encima de la matriz pero debajo de los centros
        )

    # ==========================================================
    # 4. DIBUJO DE LÍNEAS CON LEYENDAS DINÁMICAS
    # ==========================================================
    # Límites de los rangos
    for idx, (ymin, ymax) in enumerate(rangos_CF):
        if idx < len(rangos_CF) - 1:
            ax.axhline(
                y=(ymax + 0.5) * atom_size_nm,
                color="gray",
                linestyle=":",
                alpha=0.8,
                linewidth=2,
                label="Límite de Rango" if idx == 0 else "",
                zorder=3,
            )

    # CENTROS calculados
    for idx, centro in enumerate(centros_calculados):
        if centro is not None:
            y_fisico = centro
            ax.axhline(
                y=(y_fisico + 0.5) * atom_size_nm,
                color="#D32F2F",
                linestyle="-",
                alpha=0.9,
                linewidth=3,
                label=f"Centro {idx + 1} (Fila {centro})",
                zorder=3,
            )

    # Filas intermedias / muros
    for idx, fila_mid in enumerate(filas_intermedias):
        if fila_mid is not None:
            y_mid_fisico = fila_mid
            ax.axhline(
                y=(y_mid_fisico + 0.5) * atom_size_nm,
                color="#388E3C",
                linestyle="-.",
                alpha=0.9,
                linewidth=2.5,
                label=f"Muro {idx + 1} (Fila {fila_mid})",
                zorder=3,
            )

    # ==========================================================
    # 5. LÍMITES AJUSTADOS Y ESTÉTICA
    # ==========================================================
    ax.set_aspect("equal")

    ax.set_xlim(-electrode_width, size_x_nm + electrode_width)
    margen_y = 0.05
    ax.set_ylim(-margen_y, size_y_nm + margen_y)

    paso_ticks_x = 2 if size_x_nm <= 15 else (5 if size_x_nm <= 30 else 10)
    paso_ticks_y = 2 if size_y_nm <= 15 else (5 if size_y_nm <= 30 else 10)

    ax.set_xticks(np.arange(0, size_x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, size_y_nm + 1, paso_ticks_y))

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
    matriz_molde: np.ndarray,
    filename: str,
    titulo: str = "Thermal Walls Placement and Temperature",
    atom_size: float = 0.25e-09,
) -> None:
    """
    Dibuja el dispositivo en 2D. Muestra los filamentos de fondo (gris claro),
    la ubicación de los muros térmicos (colores) y los electrodos (gris oscuro).
    """
    # 1. Aplicar estilos estándar
    setup_paper_plt(plt, latex=True, scaling=3)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    # 1. CÁLCULO DE DIMENSIONES (Crecimiento en X, Clasificación en Y)
    device_size_x, device_size_y = matriz_muros.shape

    size_x_nm = device_size_x * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm
    size_y_nm = device_size_y * atom_size * 1e9  # Convertir a nm ya que siempre se representa en nm

    # ==========================================================
    # 2. CAPA DE FONDO: Matriz Molde (Filamentos)
    # ==========================================================
    # Usamos cmap "Greys" con alpha bajo para que quede como marca de agua
    ax.imshow(
        matriz_molde,
        cmap="Greys",
        origin="lower",
        alpha=0.25,
        extent=(0, size_x_nm, 0, size_y_nm),
        aspect="equal",
        zorder=2,
    )

    # ==========================================================
    # 3. CAPA DEL MURO: Temperaturas Fijas
    # ==========================================================
    # Enmascaramos los ceros (donde no hay muro) para que sean transparentes
    muros_visibles = np.ma.masked_where(matriz_muros == 0, matriz_muros)

    # Dibujamos el muro con un colormap térmico (ej: plasma)
    im_muros = ax.imshow(
        muros_visibles, cmap="plasma", origin="lower", extent=(0, size_x_nm, 0, size_y_nm), aspect="equal", zorder=3
    )

    # Añadimos la barra de color vinculada solo a la temperatura del muro
    cbar = fig.colorbar(im_muros, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fixed Boundary Temperature (K)")

    # ==========================================================
    # 4. ELECTRODOS (Estilo RepresentateState)
    # ==========================================================
    electrode_width = 0.2
    y_start = 0
    electrode_height = size_y_nm

    left_electrode = patches.Rectangle(
        (-electrode_width, y_start), electrode_width, electrode_height, color="gray", zorder=1
    )

    right_electrode = patches.Rectangle((size_x_nm, y_start), electrode_width, electrode_height, color="gray", zorder=1)

    ax.add_patch(left_electrode)
    ax.add_patch(right_electrode)

    # ==========================================================
    # 5. CONFIGURACIÓN DE EJES Y ESTÉTICA
    # ==========================================================
    ax.set_aspect("equal")

    # Ajustar límites para mostrar electrodos
    ax.set_xlim(-electrode_width, size_y_nm + electrode_width)
    margen_y = 0.05
    ax.set_ylim(-margen_y, size_y_nm + margen_y)

    # Ticks dinámicos
    paso_ticks_x = 2 if size_x_nm <= 15 else (5 if size_x_nm <= 30 else 10)
    paso_ticks_y = 2 if size_y_nm <= 15 else (5 if size_y_nm <= 30 else 10)

    ax.set_xticks(np.arange(0, size_x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, size_y_nm + 1, paso_ticks_y))

    # Textos y Títulos
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
    save_path=None,
    atom_size: float = 0.25e-9,
    filas_intermedias: list | None = None,
):
    """
    Visualiza el mapa de temperatura con superposición de materiales e isotermas alineadas.
    Permite visualizar la ubicación de los muros térmicos si se proporcionan las filas.
    """

    # 1. Aplicar los estilos globales ANTES de crear la figura
    setup_paper_plt(plt, latex=True, scaling=3)

    # 2. Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(12, 9))

    # 3. Aplicar configuración de estilo específica para los ejes
    config_ax_state(ax)

    # 4. Configuración de dimensiones y límites (Alineación perfecta)
    Ny, Nx = T_map.shape

    atom_size_nm = atom_size * 1e9  # Convertir la celda a nm
    size_x_nm = Nx * atom_size_nm  # Convertir a nm
    size_y_nm = Ny * atom_size_nm  # Convertir a nm

    extent = (0, size_x_nm, 0, size_y_nm)

    # 5. Capa base: Temperatura
    im = ax.imshow(T_map, cmap="coolwarm", origin="lower", extent=extent, aspect="equal", zorder=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Temperatura (K)")

    # 6. Capa de materiales (Overlay)
    overlay = np.zeros((Ny, Nx, 4))
    overlay[types_map == 3] = [0.2, 0.2, 0.2, 0.8]  # Electrodos: Gris oscuro
    overlay[types_map == 1] = [0.5, 0.5, 0.5, 0.4]  # Filamento: Gris claro
    ax.imshow(overlay, origin="lower", extent=extent, aspect="equal", zorder=2)

    # 7. Capa de Isotermas
    niveles = np.linspace(np.min(T_map), np.max(T_map), num_levels)
    contours = ax.contour(
        T_map, levels=niveles, colors="white", linewidths=1.5, alpha=1, origin="lower", extent=extent, zorder=3
    )
    ax.clabel(contours, fontsize=18, inline=True, fmt="%d")

    # ==========================================================
    # 8. DIBUJO DE LOS MUROS TÉRMICOS
    # ==========================================================
    if filas_intermedias is not None:
        for idx, fila_mid in enumerate(filas_intermedias):
            if fila_mid is not None:
                # Línea superior del muro (fila_mid)
                y_mid_fisico = fila_mid + 0.5
                ax.axhline(
                    y=y_mid_fisico * atom_size_nm,
                    color="#00FFFF",  # Cian brillante (contrasta muy bien con el coolwarm)
                    linestyle="--",
                    alpha=0.9,
                    linewidth=3,  # Línea gruesa para que se vea bien
                    zorder=4,  # Por encima de las isotermas y el heatmap
                    label="Thermal Wall" if idx == 0 else "",
                )

                # Línea inferior del muro (justo la fila de abajo: fila_mid - 1)
                y_mid_debajo_fisico = (fila_mid + 1) + 0.5
                ax.axhline(
                    y=y_mid_debajo_fisico * atom_size_nm,
                    color="#00FFFF",
                    linestyle="--",
                    alpha=0.9,
                    linewidth=3,
                    zorder=4,
                )
    # ==========================================================

    # 9. Estética y marcas (Ticks)
    paso_ticks_x = 2 if size_x_nm <= 15 else (5 if size_x_nm <= 30 else 10)
    paso_ticks_y = 2 if size_y_nm <= 15 else (5 if size_y_nm <= 30 else 10)

    ax.set_xticks(np.arange(0, size_x_nm + 1, paso_ticks_x))
    ax.set_yticks(np.arange(0, size_y_nm + 1, paso_ticks_y))

    ax.set_title(f"$V_{{RRAM}}$ = {voltage} V", pad=25)
    ax.set_xlabel(r"Dielectric length (\si{\nano\meter})")
    ax.set_ylabel(r"Ti electrode (\si{\nano\meter})")

    ax.set_aspect("equal")

    plt.tight_layout()

    # 10. Guardar si se especifica la ruta
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # bbox_inches='tight' es crucial para que no recorte la leyenda exterior
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close(fig)  # Cierra para liberar memoria


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
