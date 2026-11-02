"""
Estadística de la generación de vacantes en PP_set.

Recorre todas las carpetas `simulation_N` de una carpeta de resultados y calcula, para cada
simulación, a partir de los estados intermedios guardados (`set/Estado_pp_set_sim_N_paso_k.npz`),
de `Data_pp_set_N.npz`, de `Final_state_N_pp_set.npz` y de `sim_metadata_N.json`:

    1. Esperadas vs reales: vacantes que predicen los mapas de probabilidad frente a las generadas.
    2. Saturación de p: % de celdas con p = 1, p media/máx y celdas ocupadas con p > 0 en el guardado.
    3. Margen hasta el cupo: mayor salto por paso, margen mínimo y si el recorte por cupo llegó a actuar.
    4. Reparto inicial: vacantes iniciales dentro/fuera de la máscara frente a un reparto uniforme.
    5. Uniformidad de los aleatorios: histograma de todos los valores de `aleatorios_matrix` guardados
       (el U(0,1) del sorteo `aleatorios < prob_final`) y test chi-cuadrado frente a la uniforme.

Salida: tablas por consola, un Excel (.xlsx) por cálculo y gráficas PNG en `<carpeta>/estadistica_generacion/`.

Uso:
    python estadistica_generacion.py [carpeta_resultados] [--cada 500]

    Sin carpeta se usa CARPETA_POR_DEFECTO (definida al principio del script).

    --cada: cada cuántos pasos se imprime una fila en consola (los Excel llevan todos los pasos guardados).

Notas:
    - Los estados intermedios se guardan cada `pasos_guardar_estado` pasos, así que las vacantes
      esperadas se integran suponiendo que el mapa de probabilidad del paso k vale para los pasos
      hasta el siguiente guardado. Es una aproximación, exacta solo si se guarda en todos los pasos.
    - `actual_state` se guarda DESPUÉS de generar y `probabilidad_matrix` se calcula ANTES, por eso
      puede haber celdas ocupadas con p > 0 en un mismo guardado: son las recién generadas.
    - La máscara se reconstruye con RRAM.Generation.create_custom_mask, igual que en PP_set.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from RRAM.Generation import create_custom_mask
from RRAM.Representate import config_ax, guardar_figura, setup_paper_plt

FILAS_POR_BLOQUE = 10

# Carpeta que se analiza si no se pasa ninguna por argumento (p. ej. al lanzarlo con el botón Run).
# Ruta relativa a la carpeta de este script, o absoluta.
CARPETA_POR_DEFECTO = "Crecimiento_libre_T_electrodo_fija_prob_Uniforme"

# Colores de RRAM/Representate.py
COLOR_SIM = "red"  # simulación (como en plot_IV)
COLOR_REF = "black"  # referencia (como la curva experimental)
COLOR_AZUL = (0.2314, 0.2275, 0.9647)  # azul de RepresentateTwoStates
COLOR_VERDE = "#388E3C"  # verde de plot_centros_filamento_det
COLOR_CON_VECINOS = "tab:blue"  # vacantes con vecinos (esperadas y reales)
COLOR_SIN_VECINOS = "tab:red"  # vacantes aisladas (esperadas y reales)
COLOR_SIMS = "gray"  # cada simulación individual


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def buscar_simulaciones(carpeta: Path) -> list[tuple[int, Path]]:
    """Devuelve [(N, ruta)] de las carpetas simulation_N, ordenadas por N."""
    sims = []
    for ruta in carpeta.iterdir():
        m = re.fullmatch(r"simulation_(\d+)", ruta.name)
        if ruta.is_dir() and m:
            sims.append((int(m.group(1)), ruta))
    return sorted(sims)


def _cargar_npz_unico(ruta: Path) -> np.ndarray:
    """Carga el único array de un .npz guardado con np.savez(ruta, array)."""
    with np.load(ruta) as z:
        return z[z.files[0]]


def cargar_simulacion(n: int, ruta: Path) -> dict:
    """Lee metadatos, series de PP_set y la lista de estados intermedios de una simulación."""
    meta = json.loads((ruta / f"sim_metadata_{n}.json").read_text())
    params, ctes = meta["params_dict"], meta["ctes_dict"]

    with np.load(ruta / f"Data_pp_set_{n}.npz") as d:
        vacantes = d["vacantes"][:, 2].astype(int)  # vacantes al EMPEZAR cada paso
        voltaje = d["datos_sim"][:, 1]

    estados = {}
    for f in (ruta / "set").glob(f"Estado_pp_set_sim_{n}_paso_*.npz"):
        estados[int(re.findall(r"paso_(\d+)", f.name)[0])] = f
    if not estados:
        raise FileNotFoundError(f"no hay estados intermedios de PP_set en {ruta / 'set'}")

    estado_final = _cargar_npz_unico(ruta / f"Final_state_{n}_pp_set.npz")
    mascara = create_custom_mask(np.zeros(estado_final.shape), meta["centros_calculados"], ctes["grosor_filamento"])

    return {
        "n": n,
        "vacantes": vacantes,
        "voltaje": voltaje,
        "estados": dict(sorted(estados.items())),
        "estado_final": estado_final,
        "mascara": mascara,
        "cupo": int(ctes["ocupacion_max_pp_set"] * params["num_max_vacantes"]),
        "neighbor_mode": ctes.get("neighbor_mode_pp_set", "both"),
    }


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------


def mascara_vecinos(estado: np.ndarray, neighbor_mode: str) -> np.ndarray:
    """Celdas con algún vecino ocupado, con la misma regla que Generation.get_generation_probabilities_matrix."""
    ocupada = estado == 1
    vecino = np.zeros_like(ocupada)
    if neighbor_mode in ("horizontal", "both"):
        vecino[:, 1:] |= ocupada[:, :-1]
        vecino[:, :-1] |= ocupada[:, 1:]
    if neighbor_mode in ("vertical", "both"):
        vecino[1:, :] |= ocupada[:-1, :]
        vecino[:-1, :] |= ocupada[1:, :]
    return vecino


def series_por_paso(sim: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recorre los estados intermedios una sola vez y devuelve dos tablas por paso guardado:
    (esperadas vs reales, saturación de p).
    """
    pasos = list(sim["estados"])
    vac, V = sim["vacantes"], sim["voltaje"]
    filas_esp, filas_sat = [], []
    # Vacantes esperadas generadas ANTES del paso k: total, en celdas con vecino y en celdas aisladas
    esperadas_acum = esperadas_acum_con = esperadas_acum_sin = 0.0
    # Lo mismo para las reales, clasificadas según si tenían vecino al generarse
    reales_acum_con = reales_acum_sin = 0
    estado_anterior = None  # estado guardado en el paso anterior (después de generar)

    for i, k in enumerate(pasos):
        with np.load(sim["estados"][k]) as z:
            P = z["probabilidad_matrix"]
            estado = z["actual_state"]
        if P.size == 0:
            estado_anterior = estado
            continue

        positivas = P[P > 0]
        esperadas_paso = float(P.sum())  # suma de probabilidades = vacantes esperadas en el paso k

        # P se calculó con el estado ANTES de generar, pero actual_state se guarda DESPUÉS: se quitan
        # las recién generadas (ocupadas con p > 0) para clasificar los vecinos como lo hizo el código
        estado_previo = estado.copy()
        estado_previo[(P > 0) & (estado == 1)] = 0
        con_vecino = mascara_vecinos(estado_previo, sim["neighbor_mode"])
        esperadas_paso_con = float(P[con_vecino].sum())
        esperadas_paso_sin = float(P[~con_vecino].sum())
        # El último guardado puede caer un paso después del final de la serie: se usa el estado final
        vac_k = vac[k] if k < len(vac) else int(sim["estado_final"].sum())
        reales_acum = int(vac_k - vac[pasos[0]])

        # Reales generadas entre el guardado anterior y este paso (pasos intermedios, no guardados).
        # APROXIMACIÓN: se clasifican con los vecinos del estado guardado anterior; una vacante que
        # apareció junto a otra generada en el mismo intervalo cuenta como "sin vecino".
        if estado_anterior is not None:
            intermedias = (estado_previo == 1) & (estado_anterior == 0)
            vecino_anterior = mascara_vecinos(estado_anterior, sim["neighbor_mode"])
            reales_acum_con += int((intermedias & vecino_anterior).sum())
            reales_acum_sin += int((intermedias & ~vecino_anterior).sum())
        hay = positivas.size > 0

        filas_esp.append(
            {
                "sim": sim["n"],
                "paso": k,
                "V[V]": V[k] if k < len(V) else np.nan,
                "p_media": positivas.mean() if hay else 0.0,
                "p_max": positivas.max() if hay else 0.0,
                "celdas_p_pos": int(positivas.size),
                "esperadas_paso": esperadas_paso,
                "esperadas_paso_con_vecinos": esperadas_paso_con,
                "esperadas_paso_sin_vecinos": esperadas_paso_sin,
                "esperadas_acum": esperadas_acum,
                "esperadas_acum_con_vecinos": esperadas_acum_con,
                "esperadas_acum_sin_vecinos": esperadas_acum_sin,
                "reales_acum": reales_acum,
                "reales_acum_con_vecinos": reales_acum_con,
                "reales_acum_sin_vecinos": reales_acum_sin,
            }
        )
        filas_sat.append(
            {
                "sim": sim["n"],
                "paso": k,
                "V[V]": V[k] if k < len(V) else np.nan,
                "celdas_p_pos": int(positivas.size),
                "pct_p_igual_1": 100 * float((positivas == 1).mean()) if hay else 0.0,
                "p_media": positivas.mean() if hay else 0.0,
                "ocupadas_con_p_pos": int(((P > 0) & (estado == 1)).sum()),
            }
        )

        # Se supone que el mapa del paso k vale hasta el siguiente guardado
        siguiente = pasos[i + 1] if i + 1 < len(pasos) else k + 1
        esperadas_acum += esperadas_paso * (siguiente - k)
        esperadas_acum_con += esperadas_paso_con * (siguiente - k)
        esperadas_acum_sin += esperadas_paso_sin * (siguiente - k)

        # Reales generadas EN este paso: clasificación exacta, con el estado previo del propio paso
        nuevas = (P > 0) & (estado == 1)
        reales_acum_con += int((nuevas & con_vecino).sum())
        reales_acum_sin += int((nuevas & ~con_vecino).sum())
        estado_anterior = estado

    return pd.DataFrame(filas_esp), pd.DataFrame(filas_sat)


def margen_cupo(sim: dict) -> dict:
    """Saltos de vacantes por paso frente a las vacantes disponibles hasta el cupo."""
    serie = np.append(sim["vacantes"], int(sim["estado_final"].sum()))  # + estado tras el último paso
    saltos = np.diff(serie)
    margen = sim["cupo"] - serie[:-1]  # disponibles al empezar cada paso
    k_salto = int(np.argmax(saltos))
    k_lleno = int(np.nonzero(serie >= serie[-1])[0][0])
    return {
        "sim": sim["n"],
        "inicial": int(serie[0]),
        "final": int(serie[-1]),
        "celdas_mascara": int(sim["mascara"].sum()),
        "cupo": sim["cupo"],
        "max_salto": int(saltos.max()),
        "paso_max_salto": k_salto,
        "margen_min": int(margen.min()),
        "holgura_min": int((margen - saltos).min()),  # < 0 en algún paso => el recorte actuó
        "recorte_actuo": bool((saltos > margen).any() or serie.max() >= sim["cupo"]),
        "paso_final_alcanzado": k_lleno,
        "pasos_totales": len(sim["vacantes"]),
        "vacantes_decrecen": bool((saltos < 0).any()),  # en PP_set debería ser False
    }


def reparto_inicial(sim: dict) -> dict:
    """Vacantes iniciales dentro/fuera de la máscara y por bloques de filas frente a un reparto uniforme."""
    k0 = next(iter(sim["estados"]))
    with np.load(sim["estados"][k0]) as z:
        estado = z["actual_state"]
    total = int(estado.sum())
    mascara = sim["mascara"]
    filas = estado.shape[0]

    fila = {
        "sim": sim["n"],
        "paso_estado": k0,
        # El guardado del paso 0 ya incluye la generación de ese paso; es exacto si coincide con Data_pp_set
        "es_inicial_exacto": total == int(sim["vacantes"][0]),
        "total": total,
        "dentro_mascara": int(estado[mascara].sum()),
        "fuera_mascara": int(estado[~mascara].sum()),
        "fuera_esperado_uniforme": round(total * (~mascara).sum() / mascara.size, 1),
    }
    for inicio in range(0, filas, FILAS_POR_BLOQUE):
        fin = min(inicio + FILAS_POR_BLOQUE, filas) - 1
        fila[f"filas_{inicio:03d}_{fin:03d}"] = int(estado[inicio : fin + 1].sum())
    return fila


def histograma_aleatorios(sims: list[dict], bins: int = 100) -> dict | None:
    """
    Acumula el histograma de `aleatorios_matrix` de todos los estados guardados de todas las simulaciones
    (sin cargar todos los valores a la vez). Los estados sin esa clave o sin sorteo (array vacío) se saltan.
    Devuelve None si no hay ningún dato.
    """
    bordes = np.linspace(0.0, 1.0, bins + 1)
    cuentas = np.zeros(bins, dtype=np.int64)
    suma = suma2 = 0.0
    n_archivos = 0
    for sim in sims:
        for ruta in sim["estados"].values():
            with np.load(ruta) as z:
                if "aleatorios_matrix" not in z.files:
                    continue
                a = z["aleatorios_matrix"].ravel()
            if a.size == 0:
                continue
            cuentas += np.histogram(a, bins=bordes)[0]
            suma += a.sum()
            suma2 += np.square(a).sum()
            n_archivos += 1
    n = int(cuentas.sum())
    if n == 0:
        return None
    media = suma / n
    chi2, p_valor = chisquare(cuentas)  # esperado: mismas cuentas en todos los bins
    return {
        "bordes": bordes,
        "cuentas": cuentas,
        "resumen": {
            "n_archivos": n_archivos,
            "n_valores": n,
            "media": media,  # uniforme: 0.5
            "std": np.sqrt(suma2 / n - media**2),  # uniforme: 1/sqrt(12) = 0.2887
            "chi2": chi2,
            "gl": bins - 1,
            "p_valor": p_valor,
        },
    }


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------


def _figura(titulo: str, xlabel: str, ylabel: str):
    """Figura con el estilo de RRAM/Representate.py (tamaño de plot_IV y rejilla de config_ax)."""
    fig, ax = plt.subplots(figsize=(12, 9))
    config_ax(ax)
    ax.set_title(titulo, pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def _guardar(fig, ruta: Path) -> None:
    """Guarda en PNG con Representate.guardar_figura (dpi 300, bbox ajustado) y cierra la figura."""
    guardar_figura(fig, str(ruta.with_suffix("")), "png")
    plt.close(fig)


def grafica_esperadas_vs_reales(df: pd.DataFrame, ruta: Path) -> None:
    """
    Mediana entre simulaciones de vacantes acumuladas frente al voltaje, esperadas (discontinua) y
    reales (continua), con el color según la población: con vecinos y sin vecinos. Los totales (y su
    banda mín–máx) están comentados en `series`; basta descomentarlos para volver a dibujarlos.

    Con un número par de simulaciones la mediana es la media de los dos valores centrales, así que
    las reales (enteras en cada simulación) pueden salir con decimales (p. ej. 0.5).

    El voltaje de cada paso es la mediana entre simulaciones (coincide si comparten protocolo). Con
    protocolos no monótonos (trenes de pulsos) la curva vuelve sobre sí misma.
    """
    fig, ax = _figura("Cumulative generated vacancies", r"Voltage (V)", "Cumulative vacancies")
    voltaje = df.pivot_table(index="paso", columns="sim", values="V[V]").median(axis=1)
    series = (
        # Totales comentados para que se vean mejor las curvas con y sin vecinos
        # ("esperadas_acum", COLOR_REF, "--", "Expected (total)", False),
        ("esperadas_acum_con_vecinos", COLOR_CON_VECINOS, "--", "Expected, with neighbours", False),
        ("esperadas_acum_sin_vecinos", COLOR_SIN_VECINOS, "--", "Expected, isolated", False),
        # ("reales_acum", COLOR_REF, "-", "Actual (total)", True),
        ("reales_acum_con_vecinos", COLOR_CON_VECINOS, "-", "Actual, with neighbours", False),
        ("reales_acum_sin_vecinos", COLOR_SIN_VECINOS, "-", "Actual, isolated", False),
    )
    for col, color, estilo, nombre, con_banda in series:
        tabla = df.pivot_table(index="paso", columns="sim", values=col).reindex(voltaje.index)
        # Estadísticos con los ceros incluidos; se ocultan después porque la escala log no los dibuja
        mediana, minimo, maximo = (s.where(s > 0) for s in (tabla.median(axis=1), tabla.min(axis=1), tabla.max(axis=1)))
        if con_banda:
            ax.fill_between(voltaje.values, minimo, maximo, color=color, alpha=0.15, linewidth=0)
        ax.plot(voltaje.values, mediana, color=color, linestyle=estilo, linewidth=3, label=nombre)
    ax.set_yscale("log")
    ax.legend(loc="lower right", title="Median across simulations")
    _guardar(fig, ruta)


# Series comunes a las gráficas de esperadas vs reales: (columna, color, estilo, nombre)
SERIES_ESPERADAS_REALES = (
    # Totales comentados para que se vean mejor las curvas con y sin vecinos
    # ("esperadas_acum", COLOR_REF, "--", "Expected (total)"),
    ("esperadas_acum_con_vecinos", COLOR_CON_VECINOS, "--", "Expected, with neighbours"),
    ("esperadas_acum_sin_vecinos", COLOR_SIN_VECINOS, "--", "Expected, isolated"),
    # ("reales_acum", COLOR_REF, "-", "Actual (total)"),
    ("reales_acum_con_vecinos", COLOR_CON_VECINOS, "-", "Actual, with neighbours"),
    ("reales_acum_sin_vecinos", COLOR_SIN_VECINOS, "-", "Actual, isolated"),
)


def grafica_esperadas_vs_reales_panel(df: pd.DataFrame, ruta: Path, columnas: int = 2) -> None:
    """
    Panel con un gráfico por simulación: las mismas seis curvas que grafica_esperadas_vs_reales,
    pero con los valores de esa simulación (sin medianas ni bandas) frente a su propio voltaje.
    Eje y logarítmico y compartido para poder comparar simulaciones.
    """
    sims = sorted(df["sim"].unique())
    columnas = min(columnas, len(sims))
    filas = int(np.ceil(len(sims) / columnas))

    # Mismo estilo que el resto, con letra más pequeña para que quepan los subgráficos
    with plt.rc_context():
        setup_paper_plt(plt, latex=True, scaling=1.5)
        fig, axes = plt.subplots(filas, columnas, figsize=(7 * columnas, 5.5 * filas), sharey=True, squeeze=False)
        for ax, sim in zip(axes.flat, sims):
            d = df[df["sim"] == sim].sort_values("paso")
            config_ax(ax)
            for col, color, estilo, nombre in SERIES_ESPERADAS_REALES:
                valores = d[col].where(d[col] > 0)  # escala log: los ceros no se pueden dibujar
                ax.plot(d["V[V]"], valores, color=color, linestyle=estilo, linewidth=2, label=nombre)
            ax.set_yscale("log")
            ax.set_title(f"Simulation {sim}")
            ax.set_xlabel("Voltage (V)")
        for ax in axes[:, 0]:
            ax.set_ylabel("Cumulative vacancies")
        for ax in axes.flat[len(sims) :]:
            ax.set_visible(False)  # huecos sobrantes de la rejilla

        handles, etiquetas = axes.flat[0].get_legend_handles_labels()
        axes.flat[0].legend(handles, etiquetas, loc="lower right")  # leyenda común, dentro del primer gráfico
        fig.tight_layout()
        guardar_figura(fig, str(ruta.with_suffix("")), "png", dpi=200)
        plt.close(fig)


def grafica_esperadas_vs_reales_mediana_vs_media(df: pd.DataFrame, ruta: Path) -> None:
    """
    Dos gráficos para decidir entre mediana y media como resumen entre simulaciones:
      izquierda: mediana con banda mín–máx;  derecha: media.
    En ambos, esperadas (discontinua) y reales (continua) con el mismo estadístico, y cada
    simulación individual en trazo fino (solo las reales) para ver si alguna arrastra la media.
    Eje y logarítmico y compartido; el voltaje es la mediana entre simulaciones por paso.
    """
    voltaje = df.pivot_table(index="paso", columns="sim", values="V[V]").median(axis=1)
    with plt.rc_context():
        setup_paper_plt(plt, latex=True, scaling=1.5)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, squeeze=False)
        for ax, estadistico, titulo in zip(axes.flat, ("median", "mean"), ("Median and min--max band", "Mean")):
            config_ax(ax)
            for col, color, estilo, nombre in SERIES_ESPERADAS_REALES:
                tabla = df.pivot_table(index="paso", columns="sim", values=col).reindex(voltaje.index)
                es_real = estilo == "-"
                if es_real:  # una línea fina por simulación
                    for sim in tabla.columns:
                        ax.plot(voltaje.values, tabla[sim].where(tabla[sim] > 0), color=color, alpha=0.3, linewidth=0.8)
                resumen = tabla.median(axis=1) if estadistico == "median" else tabla.mean(axis=1)
                if es_real and estadistico == "median":
                    minimo, maximo = (
                        tabla.min(axis=1).where(lambda s: s > 0),
                        tabla.max(axis=1).where(lambda s: s > 0),
                    )
                    ax.fill_between(voltaje.values, minimo, maximo, color=color, alpha=0.15, linewidth=0)
                ax.plot(
                    voltaje.values,
                    resumen.where(resumen > 0),
                    color=color,
                    linestyle=estilo,
                    linewidth=2.5,
                    label=nombre,
                )
            ax.set_yscale("log")
            ax.set_title(titulo)
            ax.set_xlabel("Voltage (V)")
        axes[0, 0].set_ylabel("Cumulative vacancies")
        axes[0, 0].legend(loc="lower right")
        fig.tight_layout()
        guardar_figura(fig, str(ruta.with_suffix("")), "png", dpi=200)
        plt.close(fig)


def grafica_saturacion(df: pd.DataFrame, ruta: Path) -> None:
    """% de celdas con p > 0 que tienen p = 1, por simulación y mediana."""
    fig, ax = _figura("Probability saturation", "Step", r"Cells with $p>0$ that have $p=1$ (\%)")
    tabla = df.pivot_table(index="paso", columns="sim", values="pct_p_igual_1")
    for i, sim in enumerate(tabla.columns):
        ax.plot(tabla.index, tabla[sim], color=COLOR_SIMS, linewidth=1.5, label="Each simulation" if i == 0 else None)
    ax.plot(tabla.index, tabla.median(axis=1), color=COLOR_SIM, linewidth=3, label="Median")
    # Solo interesa la ventana en la que alguna simulación tiene p = 1 (fuera de ella todo es 0)
    activos = tabla.index[(tabla > 0).any(axis=1)]
    if len(activos):
        margen = max(50, (activos[-1] - activos[0]) // 5)
        ax.set_xlim(activos[0] - margen, activos[-1] + margen)
    ax.legend(loc="upper left")
    _guardar(fig, ruta)


def grafica_vacantes(sims: list[dict], margenes: pd.DataFrame, ruta: Path) -> None:
    """Vacantes por paso de cada simulación con el cupo y las celdas de la máscara como referencia."""
    fig, ax = _figura("Vacancies per step vs. cap", "Step", "Vacancies")
    longitud = max(len(s["vacantes"]) for s in sims)
    matriz = np.full((len(sims), longitud), np.nan)
    for i, sim in enumerate(sims):
        ax.plot(sim["vacantes"], color=COLOR_SIMS, linewidth=1.5, label="Each simulation" if i == 0 else None)
        matriz[i, : len(sim["vacantes"])] = sim["vacantes"]
    ax.plot(np.nanmedian(matriz, axis=0), color=COLOR_SIM, linewidth=3, label="Median")
    for valor, texto in ((margenes["cupo"].max(), "Cap"), (margenes["celdas_mascara"].max(), "Mask cells")):
        ax.axhline(valor, color=COLOR_REF, linewidth=2, linestyle="--")
        ax.text(longitud * 0.01, valor, f" {texto}: {int(valor)}", va="bottom")
    ax.set_ylim(top=margenes["cupo"].max() * 1.1)  # deja sitio a la etiqueta del cupo
    ax.legend(loc="center right")
    _guardar(fig, ruta)


def grafica_reparto_inicial(df: pd.DataFrame, sims: list[dict], ruta: Path) -> None:
    """Media entre simulaciones de vacantes iniciales por bloque de filas frente a un reparto uniforme."""
    fig, ax = _figura("Initial vacancy distribution", "Rows", "Initial vacancies")
    cols = [c for c in df.columns if c.startswith("filas_")]
    medias = df[cols].mean()
    filas = sims[0]["estado_final"].shape[0]
    tam_bloque = np.array([int(c.split("_")[2]) - int(c.split("_")[1]) + 1 for c in cols])
    uniforme = df["total"].mean() * tam_bloque / filas
    etiquetas = [f"{int(c.split('_')[1])}--{int(c.split('_')[2])}" for c in cols]
    x = np.arange(len(cols))
    ax.bar(x, medias.values, width=0.7, color=COLOR_AZUL, label="Observed (mean)", zorder=2)
    ax.plot(x, uniforme, color=COLOR_REF, linewidth=3, marker="o", markersize=10, label="Uniform", zorder=3)
    ax.set_xticks(x, etiquetas, rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_ylim(top=max(medias.max(), uniforme.max()) * 1.45)  # la leyenda no tapa las barras
    ax.legend(loc="upper center", ncol=2)
    _guardar(fig, ruta)


def grafica_aleatorios(h: dict, ruta: Path) -> None:
    """Histograma normalizado de los aleatorios del sorteo frente a la densidad uniforme (= 1)."""
    fig, ax = _figura("Random numbers of the generation draw", "Random number", "Density")
    bordes, cuentas = h["bordes"], h["cuentas"]
    densidad = cuentas / (cuentas.sum() * np.diff(bordes))
    ax.bar(bordes[:-1], densidad, width=np.diff(bordes), align="edge", color=COLOR_AZUL, label="Observed", zorder=2)
    ax.axhline(1.0, color=COLOR_REF, linewidth=3, label="Uniform", zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(densidad.max(), 1.0) * 1.45)  # la leyenda no tapa las barras
    ax.legend(loc="upper center", ncol=2)
    _guardar(fig, ruta)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------


def _imprimir(titulo: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * len(titulo)}\n{titulo}\n{'=' * len(titulo)}")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.float_format", "{:.3g}".format):
        print(df.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Estadística de la generación de vacantes en PP_set.")
    parser.add_argument(
        "carpeta",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / CARPETA_POR_DEFECTO,
        help=f"Carpeta que contiene las carpetas simulation_N (por defecto: {CARPETA_POR_DEFECTO})",
    )
    parser.add_argument("--cada", type=int, default=500, help="Cada cuántos pasos se imprime una fila en consola")
    args = parser.parse_args()
    print(f"Analizando {args.carpeta}")

    if not args.carpeta.is_dir():
        print(f"No existe la carpeta {args.carpeta}", file=sys.stderr)
        return 1

    sims = []
    for n, ruta in buscar_simulaciones(args.carpeta):
        try:
            sims.append(cargar_simulacion(n, ruta))
        except (FileNotFoundError, KeyError) as e:
            print(f"AVISO: simulation_{n} se omite: {e}", file=sys.stderr)
    if not sims:
        print(f"No hay simulaciones válidas en {args.carpeta}", file=sys.stderr)
        return 1

    tablas_esp, tablas_sat = [], []
    for sim in sims:
        esp, sat = series_por_paso(sim)
        tablas_esp.append(esp)
        tablas_sat.append(sat)
    esperadas = pd.concat(tablas_esp, ignore_index=True)
    saturacion = pd.concat(tablas_sat, ignore_index=True)
    margenes = pd.DataFrame([margen_cupo(s) for s in sims])
    reparto = pd.DataFrame([reparto_inicial(s) for s in sims])

    salida = args.carpeta / "estadistica_generacion"
    salida.mkdir(exist_ok=True)
    esperadas.to_excel(salida / "esperadas_vs_reales.xlsx", index=False)
    saturacion.to_excel(salida / "saturacion.xlsx", index=False)
    margenes.to_excel(salida / "margen_cupo.xlsx", index=False)
    reparto.to_excel(salida / "reparto_inicial.xlsx", index=False)

    setup_paper_plt(plt, latex=True, scaling=2.5)  # mismo estilo que plot_IV en RRAM/Representate.py
    grafica_esperadas_vs_reales(esperadas, salida / "esperadas_vs_reales.png")
    grafica_esperadas_vs_reales_panel(esperadas, salida / "esperadas_vs_reales_panel.png")
    grafica_esperadas_vs_reales_mediana_vs_media(esperadas, salida / "esperadas_vs_reales_mediana_vs_media.png")
    grafica_saturacion(saturacion, salida / "saturacion.png")
    grafica_vacantes(sims, margenes, salida / "vacantes_vs_cupo.png")
    grafica_reparto_inicial(reparto, sims, salida / "reparto_inicial.png")

    aleatorios = histograma_aleatorios(sims)
    if aleatorios is None:
        print("AVISO: ningún estado guardado contiene 'aleatorios_matrix'; se omite la uniformidad.", file=sys.stderr)
    else:
        pd.DataFrame([aleatorios["resumen"]]).to_excel(salida / "uniformidad_aleatorios.xlsx", index=False)
        grafica_aleatorios(aleatorios, salida / "uniformidad_aleatorios.png")

    for sim in sims:
        n = sim["n"]
        _imprimir(
            f"Simulación {n} · esperadas vs reales",
            esperadas[(esperadas["sim"] == n) & (esperadas["paso"] % args.cada == 0)].drop(columns="sim"),
        )
        _imprimir(
            f"Simulación {n} · saturación de p",
            saturacion[(saturacion["sim"] == n) & (saturacion["paso"] % args.cada == 0)].drop(columns="sim"),
        )
    _imprimir("Margen hasta el cupo", margenes)
    _imprimir("Reparto inicial", reparto)
    if aleatorios is not None:
        _imprimir(
            "Uniformidad de los aleatorios (uniforme: media 0.5, std 0.2887)", pd.DataFrame([aleatorios["resumen"]])
        )

    print(f"\n{len(sims)} simulaciones analizadas. Excel y gráficas en {salida}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
