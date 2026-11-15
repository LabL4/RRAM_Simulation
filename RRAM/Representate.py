"""
Figuras de la simulación: estado del dispositivo, mapas (temperatura, probabilidad, muro,
resistencias), curvas I-V y formas de onda.

Todas las figuras del dispositivo comparten ejes (nm, electrodos a los lados): lo común está en
`_electrodos` y `_ejes_dispositivo`. Las dos curvas I-V salen de `_figura_IV`.
"""

import logging
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import markers
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: Interruptor global del renderizado LaTeX de todas las figuras. Se cambia con `configurar_latex`.
#: Las funciones de dibujo lo leen al llamar a `setup_paper_plt` / `setup_plt` (sin argumento `latex`).
_USAR_LATEX = True

#: Ancho con el que se dibujan los electrodos a cada lado del dispositivo [nm].
ANCHO_ELECTRODO_NM = 0.2

#: Ciclo I-V experimental de referencia (SET y RESET), relativo a la carpeta de trabajo.
RUTA_CICLO_EXPERIMENTAL_SET = "Datos_Experimentales/Ciclos_Experimentales/Cycle_p_1000.txt"
RUTA_CICLO_EXPERIMENTAL_RESET = "Datos_Experimentales/Ciclos_Experimentales/Cycle_n_1000.txt"


# ---------------------------------------------------------------------------
# Estilo
# ---------------------------------------------------------------------------


def configurar_latex(activo: bool) -> None:
    """Activa (True, por defecto) o desactiva el renderizado LaTeX en todas las figuras siguientes."""
    global _USAR_LATEX
    _USAR_LATEX = bool(activo)


def latex_activo() -> bool:
    """True si las figuras se renderizan con LaTeX."""
    return _USAR_LATEX


def _resolver_latex(latex: bool | None) -> bool:
    """`None` → el interruptor global; un bool explícito lo anula solo para esa llamada."""
    return _USAR_LATEX if latex is None else bool(latex)


def _txt(texto: str) -> str:
    """Texto de una etiqueta: tal cual con LaTeX; sin LaTeX sustituye las unidades de siunitx por símbolos.

    Mira `text.usetex` de matplotlib (lo fija `setup_paper_plt` justo antes), así que respeta también
    el `usar_latex` explícito de una figura concreta.
    """
    if plt.rcParams["text.usetex"]:
        return texto
    return texto.replace(r"\si{\nano\meter}", "nm").replace(r"\si{\ohm}", "Ω")


def _rc_latex(plt, latex) -> None:
    """Parámetros de texto comunes a `setup_paper_plt` y `setup_plt`."""
    usar = _resolver_latex(latex)
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": usar,
            "font.family": "mathpazo" if usar else "serif",
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


def setup_paper_plt(plt, latex=None, scaling: float = 1):
    """Estilo de las figuras de publicación. Cambia los `rcParams` globales de matplotlib."""
    _rc_latex(plt, latex)

    BIGGER_SIZE = 11 * scaling
    BIGGEST_SIZE = 14 * scaling

    plt.rc("font", size=BIGGER_SIZE)
    plt.rc("axes", titlesize=BIGGEST_SIZE)
    plt.rc("axes", labelsize=BIGGEST_SIZE)
    plt.rc("xtick", labelsize=BIGGEST_SIZE)
    plt.rc("ytick", labelsize=BIGGEST_SIZE)
    plt.rc("legend", fontsize=BIGGER_SIZE)
    plt.rc("figure", titlesize=BIGGEST_SIZE)


def setup_plt(plt, latex=None, scaling=1):
    """Estilo compacto (lo usan notebooks). Cambia los `rcParams` globales de matplotlib."""
    _rc_latex(plt, latex)

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


def config_ax(ax):
    ax.grid(which="major", color="#DDDDDD", linewidth=0.8, zorder=-1)
    ax.grid(which="minor", color="#DEDEDE", linestyle=":", linewidth=0.5, zorder=-1)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def config_ax_IV(ax):
    ax.minorticks_on()


def config_ax_state(ax):
    ax.minorticks_off()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)


# ---------------------------------------------------------------------------
# Piezas comunes
# ---------------------------------------------------------------------------


def guardar_figura(fig: Figure, ruta_base: str, extension: str, dpi: int = 300) -> None:
    """
    Guarda una figura de matplotlib en el formato especificado.

    Args:
        fig (plt.Figure): La figura de matplotlib que se desea guardar.
        ruta_base (str): Ruta completa y nombre del archivo de salida (sin la extensión).
        extension (str): Formato de guardado deseado ('png', 'pdf', 'svg'); otro → png.
        dpi (int, opcional): Resolución de la imagen. Por defecto es 300.
    """
    formato = extension.lower()
    if formato not in ("png", "pdf", "svg"):
        logger.info(f"Advertencia: Formato '{formato}' no soportado. Guardando como .png")
        formato = "png"
    _savefig_ajustado(fig, f"{ruta_base}.{formato}", dpi=dpi)


def _savefig_ajustado(fig: Figure, ruta, dpi: int | None = None) -> None:
    """
    Guarda la figura con el recuadro ajustado (`bbox_inches="tight"`) sin recortar etiquetas.

    Al calcular el recuadro ajustado, matplotlib no cuenta la parte de una etiqueta de eje que
    sobresale de sus ejes: en un dispositivo estrecho y alto, «Dielectric length (nm)» es más ancha
    que los ejes y salía cortada. Se le pasan las etiquetas y los títulos como artistas extra para
    que entren enteros.
    """
    extra = list(fig.get_default_bbox_extra_artists())
    for ax in fig.axes:
        extra += [ax.xaxis.label, ax.yaxis.label, ax.title]
    kwargs = {} if dpi is None else {"dpi": dpi}
    fig.savefig(ruta, bbox_inches="tight", bbox_extra_artists=extra, **kwargs)


def _guardar_en(fig: Figure, ruta: str | None, dpi: int) -> None:
    """Guarda `fig` en `ruta` (crea su carpeta). Sin ruta no guarda."""
    if not ruta:
        return
    os.makedirs(os.path.dirname(str(ruta)) or ".", exist_ok=True)
    _savefig_ajustado(fig, ruta, dpi)


def _tamano_nm(forma: tuple, atom_size: float) -> tuple[float, float]:
    """(ancho, alto) en nm de una matriz (filas, columnas): las columnas van de electrodo a electrodo."""
    n_filas, n_cols = forma
    return n_cols * atom_size * 1e9, n_filas * atom_size * 1e9


def _paso_ticks(longitud_nm: float) -> int:
    return 2 if longitud_nm <= 15 else (5 if longitud_nm <= 30 else 10)


def _electrodos(ax, size_x_nm: float, size_y_nm: float, ancho: float = ANCHO_ELECTRODO_NM) -> None:
    """Dos rectángulos grises a los lados del dispositivo, detrás de la matriz."""
    ax.add_patch(patches.Rectangle((-ancho, 0), ancho, size_y_nm, color="gray", zorder=1))
    ax.add_patch(patches.Rectangle((size_x_nm, 0), ancho, size_y_nm, color="gray", zorder=1))


def _ejes_dispositivo(ax, size_x_nm: float, size_y_nm: float, ancho_electrodo: float | None = ANCHO_ELECTRODO_NM, ticks=None) -> None:
    """
    Ejes de una figura del dispositivo: aspecto 1:1, límites que dejan ver los electrodos
    (con `ancho_electrodo=None` no se tocan), marcas y etiquetas en nm.

    `ticks` = (marcas_x, marcas_y); por defecto, cada 2, 5 o 10 nm según el tamaño.
    """
    ax.set_aspect("equal")
    if ancho_electrodo is not None:
        ax.set_xlim(-ancho_electrodo, size_x_nm + ancho_electrodo)
        # Margen vertical mínimo para que el marco no pise las celdas de los extremos
        margen_y = 0.05
        ax.set_ylim(-margen_y, size_y_nm + margen_y)
    if ticks is None:
        ticks = (
            np.arange(0, size_x_nm + 1, _paso_ticks(size_x_nm)),
            np.arange(0, size_y_nm + 1, _paso_ticks(size_y_nm)),
        )
    ax.set_xticks(ticks[0])
    ax.set_yticks(ticks[1])
    ax.set_xlabel(_txt(r"Dielectric length (\si{\nano\meter})"))
    ax.set_ylabel(_txt(r"Ti electrode (\si{\nano\meter})"))


# ---------------------------------------------------------------------------
# Estado del dispositivo
# ---------------------------------------------------------------------------


def RepresentateState(
    matriz: np.ndarray,
    voltaje: float,
    filename: str,
    color=(0.0000, 0.0000, 0.5451),
    guardar_png: bool = True,
    atom_size: float = 0.25e-9,  # Tamaño físico de una celda en metros (ej: 0.2 nm)
    dpi: int = 600,
) -> None:
    """
    Representa el estado de una matriz de RRAM (blanco = 0, `color` = 1).

    Con `guardar_png=True` guarda `filename` tal cual (png, pdf...) a `dpi`; con False, el mismo
    nombre con extensión .svg.
    """
    cmap = ListedColormap([(1, 1, 1), color])
    # Antes de crear la figura: los textos de los ejes toman `text.usetex` al crearse.
    setup_paper_plt(plt, scaling=2)
    fig, ax = plt.subplots(figsize=(12, 10))
    config_ax_state(ax)

    size_x_nm, size_y_nm = _tamano_nm(matriz.shape, atom_size)
    logger.debug(f"Dimensiones del dispositivo: {size_x_nm:.2f} nm x {size_y_nm:.2f} nm")

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
    _electrodos(ax, size_x_nm, size_y_nm)
    ax.set_aspect("equal", adjustable="box")
    _ejes_dispositivo(ax, size_x_nm, size_y_nm)
    ax.set_title(rf"$V_{{RRAM}}$ = {voltaje:.2f} V", pad=20)

    plt.subplots_adjust(top=0.85)

    if guardar_png:
        _savefig_ajustado(fig, filename, dpi)
    else:
        _savefig_ajustado(fig, os.path.splitext(filename)[0] + ".svg")

    plt.close(fig)


def RepresentateTwoStates(
    matriz1: np.ndarray,
    matriz2: np.ndarray,
    filename: str,
    voltage: float,
    guardar_png: bool = False,
    atom_size: float = 0.25e-9,
) -> None:
    """
    Superpone dos estados en la misma figura: `matriz1` en rojo y `matriz2` en azul semitransparente.

    Con `guardar_png=True` guarda `filename` y el mismo nombre en .pdf.
    """
    # Antes de crear la figura: los textos de los ejes toman `text.usetex` al crearse.
    setup_paper_plt(plt, scaling=2.5)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    n_filas, n_cols = matriz1.shape
    size_x_nm, size_y_nm = _tamano_nm(matriz1.shape, atom_size)
    x = np.linspace(0, size_x_nm, n_cols)
    y = np.linspace(0, size_y_nm, n_filas)

    cmap1 = LinearSegmentedColormap.from_list("cmap1", [(1, 1, 1), (0.9647, 0.1725, 0.3059)], N=2)  # Rojo
    cmap2 = LinearSegmentedColormap.from_list("cmap2", [(1, 1, 1), (0.2314, 0.2275, 0.9647)], N=2)  # Azul

    ax.pcolormesh(x, y, matriz1, shading="nearest", vmin=matriz1.min(), vmax=matriz1.max(), cmap=cmap1)
    ax.pcolormesh(x, y, matriz2, shading="nearest", vmin=matriz2.min(), vmax=matriz2.max(), cmap=cmap2, alpha=0.5)

    # Electrodos algo más altos que el dispositivo
    ancho = ANCHO_ELECTRODO_NM
    ax.add_patch(patches.Rectangle((-0.3, -0.5), ancho, size_y_nm + 1, color="gray"))
    ax.add_patch(patches.Rectangle((size_x_nm, -0.5), ancho, size_y_nm + 1, color="gray"))

    ax.set_xticks(np.arange(0, size_x_nm + 1, _paso_ticks(size_x_nm)))
    ax.set_yticks(np.arange(0, size_y_nm + 1, _paso_ticks(size_y_nm)))
    ax.set_xlabel(_txt(r"Dielectric length (\si{\nano\meter})"))
    ax.set_ylabel(_txt(r"Ti electrode (\si{\nano\meter})"))
    ax.set_title(rf"V = {voltage} (V)", pad=20)

    ax.set_aspect("equal")
    ax.set_xlim(-0.3, size_x_nm + 0.3)
    ax.set_ylim(0, size_y_nm)

    plt.subplots_adjust(top=0.85)

    if guardar_png:
        _savefig_ajustado(fig, os.path.splitext(filename)[0] + ".pdf")
        _savefig_ajustado(fig, filename, 300)

    plt.close(fig)


# ---------------------------------------------------------------------------
# Curvas I-V
# ---------------------------------------------------------------------------


def _cargar_ciclo_experimental():
    """
    Carga el ciclo I-V experimental de referencia (SET y RESET).

    Returns:
        (x_set, y_set, x_reset, y_reset): voltajes y |corrientes| de cada rama, o None si faltan
        los ficheros (se avisa en el log y la figura se dibuja solo con la simulación).
    """
    rutas = [os.path.join(os.getcwd(), r) for r in (RUTA_CICLO_EXPERIMENTAL_SET, RUTA_CICLO_EXPERIMENTAL_RESET)]
    faltan = [r for r in rutas if not os.path.isfile(r)]
    if faltan:
        logger.warning(f"No se encuentra el ciclo experimental ({faltan[0]}): se dibuja solo la simulación.")
        return None

    data_set = np.loadtxt(rutas[0], skiprows=1)
    data_reset = np.loadtxt(rutas[1], skiprows=1)

    x_set = data_set[:, 0]
    y_set = data_set[:, 1]
    x_reset = data_reset[:, 0]  # TODO: comprobar si las medidas se leen ya con el signo o no
    y_reset = abs(data_reset[:, 1])
    return x_set, y_set, x_reset, y_reset


def _figura_IV(v_set, i_set, v_reset, i_reset, exponente_minimo: int, etiquetas_sim: tuple, etiquetas_exp: tuple, mostrar_experimental: bool):
    """
    Figura I-V común a `plot_IV` y `plot_IV_marcado`: ejes, curvas de la simulación y, si se pide
    y existe, el ciclo experimental. El eje Y (log) se marca desde 10^`exponente_minimo` hasta 10^-1.
    `etiquetas_*` = (leyenda del SET, leyenda del RESET); None = sin entrada en la leyenda.
    """
    setup_paper_plt(plt, scaling=2.5)

    fig, axes = plt.subplots(figsize=(12, 9))
    config_ax_IV(axes)

    axes.set_xlabel("Voltage (V)")
    axes.set_ylabel("Current (A)")
    axes.set_yscale("log")

    axes.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    axes.set_xticklabels(["-1.5", "-1.0", "-0.5", "0.0", "0.5", "1.0"])

    exponentes = range(exponente_minimo, 0)
    axes.set_yticks([10.0**e for e in exponentes])
    axes.set_yticklabels([rf"$10^{{{e}}}$" for e in exponentes])

    axes.plot(v_set, i_set, color="red", linewidth=4, label=etiquetas_sim[0])
    axes.plot(v_reset, i_reset, color="red", linewidth=4, label=etiquetas_sim[1])

    experimental = _cargar_ciclo_experimental() if mostrar_experimental else None
    if experimental is not None:
        x_set, y_set, x_reset, y_reset = experimental
        axes.plot(x_set, y_set, "black", label=etiquetas_exp[0], linewidth=2)
        axes.plot(x_reset, y_reset, "black", label=etiquetas_exp[1], linewidth=2)
    return fig, axes


def _leyenda_IV(axes) -> None:
    axes.legend(labelspacing=0.3, handletextpad=0.2, handlelength=1.0, borderaxespad=0.2, loc="lower left")


def plot_IV(
    v_set,
    i_set,
    v_reset,
    i_reset,
    num_simulation,
    titulo_figura="I-V Characteristics",
    figures_path="Results/Figures",
    extension_guardado="png",
    mostrar_experimental=True,
):
    """
    Curva I-V de la simulación, guardada como `{figures_path}/I-V_{num_simulation + 1}`.

    Parameters:
        v_set, i_set, v_reset, i_reset: voltajes y |corrientes| del SET y del RESET.
        num_simulation (int): índice 0-based de la simulación (el fichero lleva el +1).
        titulo_figura (str): título de la figura.
        figures_path (str): carpeta donde guardar la figura; se crea si no existe.
        extension_guardado (str): 'png', 'pdf' o 'svg'.
        mostrar_experimental (bool): superpone el ciclo experimental de referencia, si existe.
    """
    os.makedirs(figures_path, exist_ok=True)
    fig, axes = _figura_IV(
        v_set, i_set, v_reset, i_reset, -7, ("Simulation", None), ("Experimental", None), mostrar_experimental
    )
    axes.set_title(titulo_figura, pad=20)
    _leyenda_IV(axes)
    guardar_figura(fig, f"{figures_path}/I-V_{num_simulation + 1}", extension_guardado)
    plt.close(fig)


def plot_IV_marcado(
    v_set,
    i_set,
    v_reset,
    i_reset,
    num_simulation,
    lista_puntos,
    desplazamiento,
    figures_path="Results/Figures",
    extension_guardado="png",
    mostrar_experimental=True,
):
    """
    Curva I-V con puntos marcados, guardada como `{figures_path}/I-V_marcado_{num_simulation + 1}`.

    `lista_puntos` = {etiqueta: (V, |I|)}; `desplazamiento` = {etiqueta: (dx, factor_y)} de cada
    etiqueta respecto a su punto. El resto de argumentos, como en `plot_IV`.
    """
    os.makedirs(figures_path, exist_ok=True)
    fig, axes = _figura_IV(
        v_set, i_set, v_reset, i_reset, -9, ("SET", "RESET"), ("Set Exp.", "Reset Exp."), mostrar_experimental
    )

    for label, (xp, yp) in lista_puntos.items():
        dx, factor_y = desplazamiento.get(label, (0.02, 1.0))  # 1.0 = sin desplazamiento en y
        axes.scatter(xp, yp, color="blue", s=80, marker=markers.MarkerStyle("D"), zorder=10)
        axes.text(
            xp + dx,
            max(yp * factor_y, 1e-6),  # la etiqueta no baja de 1e-6
            label,
            fontsize=42,
            verticalalignment="bottom",
            horizontalalignment="left",
            zorder=10,
        )

    _leyenda_IV(axes)
    guardar_figura(fig, f"{figures_path}/I-V_marcado_{num_simulation + 1}", extension_guardado)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mapas sobre el dispositivo
# ---------------------------------------------------------------------------


def plot_thermal_state(T_map, types_map, voltage, num_levels=10, atom_size: float = 0.25e-9, save_path=None, dpi: int = 600):
    """
    Visualiza el mapa de temperatura con superposición de materiales e isotermas alineadas.

    Argumentos:
    - T_map: Dato de temperatura. Debe ser una matriz 2D (np.ndarray de ndim==2).
             Si es escalar o 1D se descarta y la función retorna sin generar figura.
    - types_map: Matriz de materiales (ID 1: Filamento, ID 3: Electrodo).
    - voltage: Voltaje aplicado (usado en el título).
    - num_levels: Cantidad de líneas de contorno para las isotermas.
    - save_path: (Opcional) Ruta completa con nombre de archivo para guardar la imagen.
    - dpi: Resolución de la imagen guardada.
    """
    if not isinstance(T_map, np.ndarray) or T_map.ndim < 2:
        logger.debug("plot_thermal_state: T_map no es matriz 2D — salto.")
        return

    setup_paper_plt(plt, scaling=2)
    fig, ax = plt.subplots(figsize=(16, 14))
    config_ax_state(ax)

    Ny, Nx = T_map.shape
    size_x_nm, size_y_nm = _tamano_nm(T_map.shape, atom_size)
    extent = (0, size_x_nm, 0, size_y_nm)

    # Capa base: temperatura
    im = ax.imshow(T_map, cmap="coolwarm", origin="lower", extent=extent, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Temperature (K)")

    # Capa de materiales
    overlay = np.zeros((Ny, Nx, 4))
    overlay[types_map == 3] = [0.2, 0.2, 0.2, 0.8]  # Electrodos: gris oscuro
    overlay[types_map == 1] = [0.5, 0.5, 0.5, 0.4]  # Filamento: gris claro
    ax.imshow(overlay, origin="lower", extent=extent, aspect="equal")

    # Isotermas (solo si hay gradiente real)
    if np.ptp(T_map) > 0:
        niveles = np.unique(np.linspace(np.min(T_map), np.max(T_map), num_levels))
        if len(niveles) > 1:
            contours = ax.contour(
                T_map, levels=niveles, colors="white", linewidths=1.5, alpha=1, origin="lower", extent=extent
            )
            ax.clabel(contours, fontsize=18, inline=True, fmt="%d")

    # El mapa ya incluye las columnas de electrodo: no se dibujan aparte ni se tocan los límites
    _ejes_dispositivo(ax, size_x_nm, size_y_nm, ancho_electrodo=None)
    ax.set_title(f"$V_{{RRAM}}$ = {voltage:.2f} V", pad=25)

    plt.tight_layout()
    _guardar_en(fig, save_path, dpi)
    plt.close(fig)


def RepresentateHeatmap(
    matriz: np.ndarray,
    voltaje: float,
    titulo: str = "Heatmap",
    filename: str | None = None,
    cmap_name: str = "hot",
    label_colorbar: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
    electrode_width: float = ANCHO_ELECTRODO_NM,
    cero_blanco: bool = True,
    atom_size: float = 0.25e-9,
    dpi: int = 300,
) -> None:
    """
    Representa una matriz de valores continuos (ej: probabilidades, temperaturas).

    Args:
        matriz: Matriz 2D (Ny, Nx), sin columnas de electrodo (se dibujan aparte).
        voltaje: Voltaje aplicado (usado en el título).
        titulo: Título base de la figura.
        filename: Ruta completa para guardar. Si None, no se guarda.
        cmap_name: Nombre del colormap de matplotlib.
        label_colorbar: Etiqueta de la barra de colores.
        vmin, vmax: Límites del colormap. Si None, se calculan automáticamente.
        electrode_width: Ancho visual de los electrodos en nm.
        cero_blanco: Si True, los ceros exactos se enmascaran (color blanco).
        atom_size: Tamaño físico de cada celda [m].
        dpi: Resolución de la imagen guardada.
    """
    # Antes de crear la figura: los textos de los ejes toman `text.usetex` al crearse.
    setup_paper_plt(plt, scaling=2.5)
    fig, ax = plt.subplots(figsize=(9, 15))
    config_ax_state(ax)

    size_x_nm, size_y_nm = _tamano_nm(matriz.shape, atom_size)

    cmap = plt.get_cmap(cmap_name).copy()
    if cero_blanco:
        cmap.set_bad(color="white")
        matriz_dibujo = np.ma.masked_where(matriz == 0, matriz)
    else:
        matriz_dibujo = matriz

    im = ax.imshow(
        matriz_dibujo,
        cmap=cmap,
        vmin=vmin if vmin is not None else matriz.min(),
        vmax=vmax if vmax is not None else matriz.max(),
        extent=(0, size_x_nm, 0, size_y_nm),
        origin="lower",
        interpolation="nearest",
        aspect="equal",
        zorder=2,
    )
    _electrodos(ax, size_x_nm, size_y_nm, electrode_width)
    _ejes_dispositivo(ax, size_x_nm, size_y_nm, electrode_width)
    ax.set_title(rf"{titulo} ($V_{{RRAM}}$ = {voltaje} V)", pad=20)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label_colorbar:
        cbar.set_label(label_colorbar)

    plt.subplots_adjust(top=0.85)
    if filename:
        _savefig_ajustado(fig, filename, dpi)
    plt.close(fig)


def plot_centros_filamento_det(
    matriz_state: np.ndarray,
    rangos_CF: list,
    centros_calculados: list,
    filas_intermedias: list,
    filename: str,
    atom_size: float = 0.25e-9,
    dpi: int = 300,
) -> None:
    """
    Estado con los centros de filamento calculados, los límites de cada banda y las filas donde se
    colocan los muros térmicos, sobre una cuadrícula horizontal de una celda.
    """
    # Antes de crear la figura: los textos de los ejes toman `text.usetex` al crearse.
    setup_paper_plt(plt, scaling=2.5)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    size_x_nm, size_y_nm = _tamano_nm(matriz_state.shape, atom_size)
    atom_size_nm = atom_size * 1e9

    ax.imshow(matriz_state, cmap="Blues", origin="lower", aspect="equal", extent=(0, size_x_nm, 0, size_y_nm), zorder=2)
    _electrodos(ax, size_x_nm, size_y_nm)

    # Cuadrícula horizontal: una línea por borde de celda, en todo el alto del dispositivo
    for y_grid in np.arange(0, size_y_nm + atom_size_nm, atom_size_nm):
        ax.axhline(y=y_grid, color="black", linestyle="--", alpha=0.25, linewidth=0.8, zorder=2.5)

    # Límites de los rangos
    for idx, (_, ymax) in enumerate(rangos_CF):
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

    # Centros calculados
    for idx, centro in enumerate(centros_calculados):
        if centro is not None:
            ax.axhline(
                y=(centro + 0.5) * atom_size_nm,
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
            ax.axhline(
                y=(fila_mid + 0.5) * atom_size_nm,
                color="#388E3C",
                linestyle="-.",
                alpha=0.9,
                linewidth=2.5,
                label=f"Muro {idx + 1} (Fila {fila_mid})",
                zorder=3,
            )

    _ejes_dispositivo(ax, size_x_nm, size_y_nm)
    ax.set_title("Centros de Filamento y Límites", pad=20)

    # Leyenda fuera de la gráfica a la derecha
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.95)

    plt.subplots_adjust(top=0.85)
    _guardar_en(fig, filename, dpi)
    plt.close(fig)


def plot_muro_termico(
    matriz_muros: np.ndarray,
    matriz_molde: np.ndarray,
    filename: str,
    titulo: str = "Thermal Walls Placement and Temperature",
    atom_size: float = 0.25e-09,
    dpi: int = 300,
) -> None:
    """
    Dibuja el dispositivo en 2D. Muestra los filamentos de fondo (gris claro),
    la ubicación de los muros térmicos (colores) y los electrodos (gris oscuro).
    """
    setup_paper_plt(plt, scaling=2.5)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    size_x_nm, size_y_nm = _tamano_nm(matriz_muros.shape, atom_size)
    extent = (0, size_x_nm, 0, size_y_nm)

    # Fondo: filamentos como marca de agua
    ax.imshow(matriz_molde, cmap="Greys", origin="lower", alpha=0.25, extent=extent, aspect="equal", zorder=2)

    # Muro: los ceros (donde no hay muro) se enmascaran para que sean transparentes
    muros_visibles = np.ma.masked_where(matriz_muros == 0, matriz_muros)
    im_muros = ax.imshow(muros_visibles, cmap="plasma", origin="lower", extent=extent, aspect="equal", zorder=3)

    cbar = fig.colorbar(im_muros, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fixed Boundary Temperature (K)")

    _electrodos(ax, size_x_nm, size_y_nm)
    _ejes_dispositivo(ax, size_x_nm, size_y_nm)
    ax.set_title(titulo, pad=20)

    plt.subplots_adjust(top=0.85)
    _guardar_en(fig, filename, dpi)
    plt.close(fig)


def plot_mapa_resistencias(
    mapa_resistencias: np.ndarray,
    actual_state: np.ndarray,
    filename: str,
    titulo: str = "Filament Cell Resistance Map",
    atom_size: float = 0.25e-09,
    dpi: int = 300,
) -> None:
    """
    Dibuja el dispositivo en 2D mostrando la RESISTENCIA de cada celda del filamento.

    Análogo eléctrico de `plot_muro_termico`: fondo con las vacantes en gris claro,
    encima el mapa de resistencias en color y los electrodos a los lados.

    Las celdas que no son filamento llevan `inf` en el mapa (ver
    `conduccion.mapa_resistencias`) y se enmascaran, de modo que solo se colorea
    el filamento conductor. Con `alpha_T` distinto de cero el color revela dónde el
    filamento está más caliente y por tanto es más resistivo.

    Args:
        mapa_resistencias (np.ndarray): Mapa de resistencias por celda [Ohm], marco
            interior (sin columnas de electrodo), con `inf` fuera del filamento.
        actual_state (np.ndarray): Estado del dispositivo, para la marca de agua.
        filename (str): Ruta del archivo de salida.
        titulo (str): Título de la figura.
        atom_size (float): Tamaño de celda [m].
        dpi (int): Resolución de la imagen guardada.
    """
    setup_paper_plt(plt, scaling=2.5)
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax_state(ax)

    size_x_nm, size_y_nm = _tamano_nm(mapa_resistencias.shape, atom_size)
    extent = (0, size_x_nm, 0, size_y_nm)

    # Fondo: estado del dispositivo (vacantes)
    if actual_state is not None and actual_state.ndim == 2 and actual_state.shape == mapa_resistencias.shape:
        ax.imshow(actual_state, cmap="Greys", origin="lower", alpha=0.25, extent=extent, aspect="equal", zorder=2)

    # Resistencias: las celdas sin filamento llevan inf y se enmascaran
    resistencias_visibles = np.ma.masked_where(~np.isfinite(mapa_resistencias), mapa_resistencias)
    im_resistencias = ax.imshow(
        resistencias_visibles, cmap="cividis", origin="lower", extent=extent, aspect="equal", zorder=3
    )

    cbar = fig.colorbar(im_resistencias, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(_txt(r"Cell resistance (\si{\ohm})"))

    _electrodos(ax, size_x_nm, size_y_nm)

    # Con aspect="equal" un dispositivo estrecho y alto deja el eje X muy comprimido, así que el
    # paso se elige para no superar un número de marcas por eje.
    def _paso(longitud_nm: float, max_marcas: int) -> int:
        for paso in (1, 2, 5, 10, 20, 50):
            if longitud_nm / paso <= max_marcas:
                return paso
        return 100

    ticks = (
        np.arange(0, size_x_nm + 1, _paso(size_x_nm, max_marcas=3)),  # eje corto: pocas marcas
        np.arange(0, size_y_nm + 1, _paso(size_y_nm, max_marcas=8)),
    )
    _ejes_dispositivo(ax, size_x_nm, size_y_nm, ticks=ticks)
    ax.set_title(titulo, pad=20)

    plt.subplots_adjust(top=0.85)
    _guardar_en(fig, filename, dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Formas de onda
# ---------------------------------------------------------------------------

#: Máximo de transiciones que `plot_Vt_It` marca; por encima no marca ninguna.
MAX_TRANSICIONES_MARCADAS = 10


def plot_Vt_It(
    t: np.ndarray,
    v: np.ndarray,
    i: np.ndarray,
    filename: str,
    transiciones: list | None = None,
    titulo: str = "",
    usar_latex: bool | None = None,
):
    """
    Dibuja V(t) e |I|(t) en dos paneles apilados con eje temporal común, para
    validar formas de onda no lineales (p.ej. rampa + tramo constante por
    compliance de corriente).

    Args:
        t: Vector de tiempos [s] (columna 0 de datos_sim).
        v: Vector de voltajes aplicados [V] (columna 1 de datos_sim).
        i: Vector de corrientes [A] (columna 2 de datos_sim).
        filename: Ruta completa del archivo de salida (con extensión).
        transiciones: Lista de dicts de protocolo_voltaje → ejecucion → {fase} de la metadata
            ({"k": paso, "V": voltaje, "condicion": nombre}); cada una se marca
            con una línea vertical discontinua en su tiempo t[k].
        titulo: Título global de la figura.
        usar_latex: Renderizado LaTeX de los textos. None (por defecto): el interruptor global
            (`configurar_latex`); un bool lo anula solo para esta figura.
    """
    setup_paper_plt(plt, latex=usar_latex, scaling=1.5)

    fig, (ax_v, ax_i) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    config_ax(ax_v)
    config_ax(ax_i)

    ax_v.plot(t, v, color="tab:blue", linewidth=2)
    ax_v.set_ylabel("Voltage (V)")
    if titulo:
        ax_v.set_title(titulo, pad=15)

    ax_i.plot(t, np.abs(i), color="tab:red", linewidth=2)
    ax_i.set_yscale("log")
    ax_i.set_ylabel("|Current| (A)")
    ax_i.set_xlabel("Time (s)")

    # Marcar las transiciones de la forma de onda (si las hay). Con muchas (un tren
    # de pulsos) las líneas y etiquetas tapan la curva: no se marca ninguna.
    transiciones = transiciones or []
    if len(transiciones) > MAX_TRANSICIONES_MARCADAS:
        transiciones = []
    for n, trans in enumerate(transiciones):
        k = int(trans.get("k", -1))
        if not (0 <= k < len(t)):
            continue
        etiqueta = f"{trans.get('condicion', '?')} (V={trans.get('V', float('nan')):.3f})"
        for ax in (ax_v, ax_i):
            ax.axvline(t[k], color="gray", linestyle="--", linewidth=1.5)
        # Altura alterna: dos transiciones próximas (p. ej. entrada y salida de una
        # meseta corta) tendrían sus etiquetas superpuestas.
        ax_v.annotate(
            etiqueta,
            xy=(t[k], v[k] if k < len(v) else 0),
            xytext=(5, 5 + 16 * (n % 2)),
            textcoords="offset points",
            fontsize=9,
            color="gray",
        )

    plt.tight_layout()
    _guardar_en(fig, filename, 300)
    plt.close(fig)
