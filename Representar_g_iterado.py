from matplotlib.ticker import FixedLocator
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import numpy as np
import sys
import os

import warnings

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in log10",
    category=RuntimeWarning,
    module="matplotlib.ticker",
)

# --- Ajuste de sys.path para tu módulo RRAM ---
ruta_raiz = os.getcwd() + "/"
print("Ruta raiz del proyecto:", ruta_raiz)

sys.path.append(ruta_raiz)
base_path = Path("c:/Users/jimdo/Documents/GitHub/RRAM_Simulation")
if str(base_path) not in sys.path:
    sys.path.append(str(base_path))

# Ahora importa el módulo
from RRAM.Representate import config_ax, setup_plt  # noqa: E402
from RRAM import io_manager as io  # noqa: E402

# Inicializa el estilo global
setup_plt(plt, latex=True, scaling=2)


# — flags al principio —
GENERATE_INDIVIDUAL = True
GENERATE_PANEL = True

# --- filtro para los valores únicos de g ---
# set: nunique > 2  (mínimo 3 valores distintos)
# reset: nunique > 3 (mínimo 4 valores distintos)
set_thresh = 3
reset_thresh = 1


def compute_weighted_mean(
    g_df: pd.DataFrame, scale: float, exponent: float = 5.0
) -> pd.Series:
    """
    Calcula la media ponderada de cada fila de g_df enfatizando las columnas
    con más cambios mediante una potencia >1.

    Parámetros
    ----------
    g_df : pd.DataFrame
        DataFrame con las series de g (filas = tiempos, columnas = filas de tu sim).
    scale : float
        Factor de escala (p.ej. 10/40) para convertir a unidades finales.
    exponent : float
        Exponente >1 para elevar el conteo de cambios antes de normalizar.

    Retorna
    -------
    pd.Series
        Media ponderada escalada para cada instante de tiempo.
    """
    # 1) Conteo de cambios por columna
    changes = g_df.diff(axis=0).ne(0).sum(axis=0)

    # 2) Transformar por potencia y normalizar
    w = changes.pow(exponent)
    try:
        # Normalizamos para que sumen 1
        w = w / w.sum()
    except ZeroDivisionError:
        # Fallback: si no hay columnas o todos ceros
        w = pd.Series(0.0, index=changes.index)

    # 3) Media ponderada fila a fila y escalado
    weighted_mean = g_df.mul(w, axis=1).sum(axis=1) * scale
    return weighted_mean


def weighted_mean_by_changes(df: pd.DataFrame) -> pd.Series:
    """
    Calcula la media ponderada fila a fila de un DataFrame,
    donde el peso de cada columna es directamente el número de cambios de valor.

    Args:
        df (pd.DataFrame): Cada columna representa una serie temporal.

    Returns:
        pd.Series con la media ponderada fila a fila.
    """
    # 1. Calcular cambios por columna (comparando fila con fila+1)
    changes = (df.diff().fillna(0) != 0).sum()
    print(f"   • Cambios por columna: {changes.values}")

    # 2. Normalizar pesos para que sumen 1
    if changes.sum() > 0:
        weights = changes / changes.sum()
    else:
        weights = pd.Series(
            1.0 / len(changes), index=changes.index
        )  # todos iguales si no hay cambios

    # 3. Calcular media ponderada fila a fila
    weighted_mean = (df * weights).sum(axis=1)

    return weighted_mean


# --- Carpeta destino de todas las figuras ---
figures_dir = base_path / "Results" / "Figures" / "Figures_g_evolution"
figures_dir.mkdir(parents=True, exist_ok=True)

# --- Carpeta destino para archivos CSV exportados ---
csv_dir = base_path / "Results" / "CSV_Exports"
csv_dir.mkdir(parents=True, exist_ok=True)

# --- Listar carpetas de simulación bajo Results/ ---
results_dir = base_path / "Results"
sim_dirs = sorted(
    d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("simulation_")
)

print(f"🔄 Encontradas {len(sim_dirs)} simulaciones: {[d.name for d in sim_dirs]}")

# --- Escala para g (nm) ---
scale = 10 / 40

# --- Bucle principal ---
for sim_dir in sim_dirs:
    num = sim_dir.name.split("_")[1]
    print(f"\n▶️ Procesando simulación {num} …")

    # 1) Construir rutas de datos
    paths = {
        "g_set": sim_dir / "set" / f"g_set_{num}.txt",
        "csv_set": sim_dir / "set" / f"Resultados_set_{num}.csv",
        "g_res": sim_dir / "reset" / f"g_reset_{num}.txt",
        "csv_res": sim_dir / "reset" / f"Resultados_reset_{num}.csv",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        print(f"⚠️ Sim {num}: faltan {missing}, omitiendo.")
        continue

    # 2) Cargar datos
    data_g_set = io.leer_txt_as_csv(str(paths["g_set"]), header=None)  # type: ignore
    data_set = io.leer_csv(str(paths["csv_set"]))
    data_g_reset = io.leer_txt_as_csv(str(paths["g_res"]), header=None)  # type: ignore
    data_res = io.leer_csv(str(paths["csv_res"]))

    # 3) DataFrames
    g_set_df = pd.DataFrame(data_g_set)
    g_res_df = pd.DataFrame(data_g_reset)

    # 4) Filtrar columnas para eliminar las que no cambian
    g_filt_set = g_set_df.loc[:, g_set_df.nunique(dropna=False) > set_thresh]
    g_filt_res = g_res_df.loc[:, g_res_df.nunique(dropna=False) > reset_thresh]

    g_filt_set = g_filt_set * scale  # Escalado aquí para evitar errores en el filtro
    g_filt_res = g_filt_res * scale  # Escalado aquí para evitar errores en el filtro

    print("DataFrame g_filt_set completo:")

    print(f"   • SET: antes={g_set_df.shape[1]} cols, después={g_filt_set.shape[1]}")
    print(f"   • RESET: antes={g_res_df.shape[1]} cols, después={g_filt_res.shape[1]}")

    # g_mean_set = compute_weighted_mean(g_filt_set, scale)
    # g_mean_res = compute_weighted_mean(g_filt_res, scale)
    g_mean_set = weighted_mean_by_changes(g_filt_set)
    g_mean_res = weighted_mean_by_changes(g_filt_res)

    # Crear DataFrame que combine voltaje y valores g para reset
    df_g_reset = pd.DataFrame({"Voltaje": data_res["Voltaje [V]"], "g": g_mean_res})

    # Crear DataFrame que combine voltaje y valores g para set
    df_g_set = pd.DataFrame({"Voltaje": data_set["Voltaje [V]"], "g": g_mean_set})

    # De estos valores de g, tengo que filtrar los que están entre 0.2 y 0.7 para el reset y entre 0.2 y 1.7 para el set
    # df_g_reset_filtrado = df_g_reset[
    #     (df_g_reset["g"] >= 0.2) & (df_g_reset["g"] <= 0.7)
    # ]
    # df_g_set_filtrado = df_g_set[(df_g_set["g"] >= 0.2) & (df_g_set["g"] <= 1.7)]

    # # Paso a csv cada DataFrame ya filtrado entre los valores de g
    # csv_dir.mkdir(parents=True, exist_ok=True)
    # csv_reset_path = csv_dir / f"g_reset_filtrado_sim_{num}.csv"
    # csv_set_path = csv_dir / f"g_set_filtrado_sim_{num}.csv"

    # df_g_reset_filtrado.to_csv(csv_reset_path, index=False)
    # df_g_set_filtrado.to_csv(csv_set_path, index=False)

    # De estos valores de g, tengo que filtrar los que están entre 0.2 y 0.7 para el reset y entre 0.2 y 1.7 para el set
    df_g_reset_filtrado = df_g_reset
    df_g_set_filtrado = df_g_set

    if GENERATE_INDIVIDUAL:
        print(f"   • Generando figuras individuales para sim {num} …")
        # === Figuras individuales ===
        # 1) g completa – set
        fig1, ax1 = plt.subplots(figsize=(12, 9))
        config_ax(ax1)

        yticks = np.arange(40, -1, -5) * scale
        ax1.yaxis.set_major_locator(FixedLocator(yticks))  # type: ignore

        for col in g_filt_set.columns:
            ax1.plot(data_set["Voltaje [V]"], g_filt_set[col], label=f"Fila {col}")

        ax1.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ set (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g completa (set)",
        )
        figure_path = str(figures_dir) + "/" + f"g_completa_set_{num}.png"
        ax1.legend(fontsize="x-small", loc="best")
        fig1.savefig(figure_path, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig1)

        # 2) g medio – set
        fig2, ax2 = plt.subplots(figsize=(12, 9))
        config_ax(ax2)
        ax2.scatter(
            df_g_set_filtrado["Voltaje"],
            df_g_set_filtrado["g"],
            color="C1",
            label="g medio",
        )
        ax2.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ set (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g medio (set)",
        )
        figure_path = str(figures_dir) + "/" + f"g_medio_set_{num}.png"
        # ax2.legend(fontsize="small", loc="best")
        fig2.savefig(figure_path, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig2)

        # 3) g completa – reset
        fig3, ax3 = plt.subplots(figsize=(12, 9))
        config_ax(ax3)
        yticks = np.arange(40, -1, -5) * scale
        ax3.yaxis.set_major_locator(FixedLocator(yticks))  # type: ignore
        for col in g_filt_res.columns:
            ax3.scatter(data_res["Voltaje [V]"], g_filt_res[col], label=f"Fila {col}")
        ax3.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ reset (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g completa (reset)",
        )
        figure_path = str(figures_dir) + "/" + f"g_completa_reset_{num}.png"
        ax3.legend(fontsize="small", loc="best")
        fig3.savefig(figure_path, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig3)

        # 4) g medio – reset
        fig4, ax4 = plt.subplots(figsize=(12, 9))
        config_ax(ax4)
        ax4.scatter(
            df_g_reset_filtrado["Voltaje"],
            df_g_reset_filtrado["g"],
            color="C3",
            label="g medio",
        )
        ax4.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ reset (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g medio (reset)",
        )
        figure_path = str(figures_dir) + "/" + f"g_medio_reset_{num}.png"
        # ax4.legend(fontsize="small", loc="best")
        fig4.savefig(figure_path, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig4)

    if GENERATE_PANEL:
        print("   • Generando panel 2x2 con las 4 gráficas …")
        # === Panel 2×2 con las 4 gráficas ===
        fig_panel, axes = plt.subplots(2, 2, figsize=(16, 12))
        ax_cset, ax_mset, ax_crest, ax_mreset = axes.flat

        # g completa – set
        config_ax(ax_cset)
        yticks = np.arange(40, -1, -5) * scale
        ax_cset.yaxis.set_major_locator(FixedLocator(yticks))  # type: ignore
        for col in g_filt_set.columns:
            ax_cset.scatter(
                data_set["Voltaje [V]"], g_filt_set[col], label=f"Fila {col}"
            )
        ax_cset.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ set (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g completa (set)",
        )
        ax_cset.legend(fontsize="x-small", loc="best")

        # g medio – set
        config_ax(ax_mset)
        ax_mset.scatter(
            df_g_set_filtrado["Voltaje"],
            df_g_set_filtrado["g"],
            color="C1",
            label="g medio",
        )
        ax_mset.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ set (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g medio (set)",
        )
        # ax_mset.legend(fontsize="x-small", loc="best")

        # g completa – reset
        config_ax(ax_crest)
        for col in g_filt_res.columns:
            ax_crest.scatter(
                data_res["Voltaje [V]"], g_filt_res[col], label=f"Fila {col}"
            )
        ax_crest.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ reset (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g completa (reset)",
        )
        ax_crest.legend(fontsize="x-small", loc="best")

        # g medio – reset
        config_ax(ax_mreset)
        ax_mreset.scatter(
            df_g_reset_filtrado["Voltaje"],
            df_g_reset_filtrado["g"],
            color="C3",
            label="g medio",
        )
        ax_mreset.set(
            xlabel=r"Voltage (\SI{}{\volt})",
            ylabel=r"$g$ reset (\SI{}{\nano\meter})",
            # title=f"Sim {num}: g medio (reset)",
        )
        # ax_mreset.legend(fontsize="x-small", loc="best")

        panel_path = figures_dir / f"g_simulacion_{num}.png"
        fig_panel.savefig(panel_path, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig_panel)

        print(f"✔️ Sim {num}: figuras individuales y panel guardados en {figures_dir}")
