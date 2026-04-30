"""
Postprocesado y plot a partir de datos en disco.

Lee `Data_*.npz` y `sim_metadata_{N}.json` que dejó `run_cycle` en
`Results/simulation_{N}/`. NO requiere haber ejecutado el ciclo en la misma
sesión: replotear es independiente de la ejecución.

Uso programático:

    from RRAM.plot_results import plot_results, plot_estados
    plot_results(num_simulation=3)
    plot_estados(plot_type="state", phases=["pp_set", "sp_set"])

O vía CLI: `python -m RRAM plot 3`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from .iv_analysis import simulation_IV
from .persistence import load_metadata, save_metadata
from .run_cycle import DESPLAZAMIENTO_IV_DEFAULT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes de fase
# ---------------------------------------------------------------------------

FASES_SET = ("pp_set", "sp_set")
FASES_RESET = ("pp_reset", "sp_reset")
TODAS_FASES = FASES_SET + FASES_RESET

# Tipos de plot soportados → subcarpeta dentro de Figures/
TIPO_SUBCARPETA: dict[str, str] = {
    "state": "state",
    "thermal": "thermal_state",
    "probability": "probability",
    "muro": "muro_termico",
}

PlotType = Literal["state", "thermal", "probability", "muro"]

TODOS_PLOT_TYPES: tuple[PlotType, ...] = ("state", "thermal", "probability", "muro")


def plot_results(
    num_simulation: int,
    results_dir: Path | str = "Results",
    figures_dir: Optional[Path | str] = None,
    desplazamiento: Optional[dict] = None,
    skip_failed: bool = True,
) -> bool:
    """
    Genera las figuras I-V de una simulación leyendo todo del disco.

    Args:
        num_simulation: Índice usado al ejecutar (offset +1 del run_cycle: el
            N+1 que aparece en `Results/simulation_{N+1}/`).
        results_dir: Carpeta raíz con `simulation_{N}/`.
        figures_dir: Si se omite, se usa `simulation_{N}/Figures` (la carpeta
            propia de la simulación). Antes el código creaba `simulation_{N-1}/
            Figures`, lo que dejaba `simulation_0/` vacía cuando N=1.
        desplazamiento: Anotaciones de la curva I-V. Si se omite, se usa el
            default de `run_cycle.DESPLAZAMIENTO_IV_DEFAULT` (no se persiste
            en metadata: es preferencia de plot, no estado de simulación).
        skip_failed: Si True (default) y la metadata indica que la simulación
            no completó (`status != "completed"`) o no hay roturas, se loguea
            una advertencia y se devuelve False sin intentar dibujar — evita
            el `KeyError: 0` de `simulation_IV` cuando faltan datos del reset.

    Returns:
        True si se generaron figuras; False si la simulación se saltó.

    Raises:
        FileNotFoundError: Si no se encuentra `sim_metadata_{N}.json`.
    """
    results_dir = Path(results_dir)
    simulation_path = results_dir / f"simulation_{num_simulation}"

    meta = load_metadata(simulation_path, num_simulation=num_simulation)

    status = (meta.extra or {}).get("status", "unknown")
    error_msg = (meta.extra or {}).get("error")

    # ----- Guard: solo replotear sims completas con roturas registradas -----
    if skip_failed:
        if status != "completed":
            logger.warning(
                f"Sim {num_simulation} no completada (status={status!r}). "
                f"Salto plot. Causa: {error_msg or 'sin info'}"
            )
            return False
        if not meta.roturas_dict:
            logger.warning(
                f"Sim {num_simulation} sin roturas registradas; el plot I-V "
                f"requiere al menos rotura 0. Salto plot. "
                f"Probablemente la fase de reset no llegó a destruir filamentos."
            )
            return False
        if 0 not in meta.roturas_dict:
            logger.warning(
                f"Sim {num_simulation}: roturas_dict no contiene la clave 0 "
                f"(claves: {list(meta.roturas_dict.keys())}). Salto plot."
            )
            return False

    # Las figuras viven dentro de la propia carpeta de simulación.
    if figures_dir is None:
        figures_dir = simulation_path / "Figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    desp = desplazamiento or DESPLAZAMIENTO_IV_DEFAULT

    logger.info(
        f"Replot · sim={num_simulation} · status={status} · "
        f"V_perco={meta.voltaje_percolacion:.4f}V · "
        f"creaciones={len(meta.creaciones_dict)} · roturas={len(meta.roturas_dict)}"
    )

    simulation_IV(
        num_simulation=num_simulation,
        figures_path=figures_dir,
        simulation_path=simulation_path,
        desplazamiento=desp,
        voltaje_percolacion=meta.voltaje_percolacion,
        roturas_dict=meta.roturas_dict,
    )
    return True


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _extraer_fase_y_paso(nombre: str) -> tuple[str, int] | None:
    """
    Extrae fase y paso del nombre de un archivo Estado.

    Formato esperado: ``Estado_{fase}_sim_{N}_paso_{paso}.npz``

    Returns:
        (fase, paso) o None si el nombre no coincide.
    """
    m = re.match(r"Estado_(?P<fase>pp_set|sp_set|pp_reset|sp_reset)_sim_\d+_paso_(?P<paso>\d+)\.npz$", nombre)
    if m is None:
        return None
    return m.group("fase"), int(m.group("paso"))


def _cargar_voltaje_por_fase(
    simulation_path: Path, num_simulation: int
) -> dict[str, np.ndarray]:
    """
    Carga el array datos_sim (col 1 = voltaje) por fase desde los Data_*.npz.

    Returns:
        Dict {fase: array_voltajes} donde array_voltajes[paso] = voltaje aplicado.
    """
    voltajes: dict[str, np.ndarray] = {}
    for fase in TODAS_FASES:
        data_file = simulation_path / f"Data_{fase}_{num_simulation}.npz"
        if data_file.exists():
            d = np.load(data_file, allow_pickle=True)
            if "datos_sim" in d:
                datos = d["datos_sim"]
                if datos.ndim == 2 and datos.shape[1] >= 2:
                    voltajes[fase] = datos[:, 1]
    return voltajes


def _temperatura_escalar(temp_data: np.ndarray) -> float:
    """Devuelve temperatura máxima independientemente de si es escalar o mapa 2D."""
    if temp_data.ndim == 0:
        return float(temp_data.item())
    return float(np.max(temp_data))


def _construir_types_map(actual_state: np.ndarray, t_shape: tuple[int, int]) -> np.ndarray:
    """
    Construye types_map compatible con plot_thermal_state.

    El mapa de temperatura tiene 2 columnas extra (electrodos, 1 a cada lado).
    types_map: valor 1 = vacante/filamento, 3 = electrodo, 0 = dieléctrico.
    """
    Ny, Nx_T = t_shape
    types_map = np.zeros((Ny, Nx_T), dtype=np.int32)
    # Electrodos en primera y última columna
    types_map[:, 0] = 3
    types_map[:, -1] = 3
    # Interior: mapeamos actual_state (columnas 1..Nx_T-2)
    Nx_dev = actual_state.shape[1]
    col_ini = (Nx_T - 2 - Nx_dev) // 2 + 1  # centrado; habitualmente = 1
    col_fin = col_ini + Nx_dev
    types_map[:, col_ini:col_fin] = actual_state
    return types_map


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def plot_estados(
    plot_types: list[PlotType] | None = None,
    phases: list[str] | None = None,
    results_dir: Path | str = "Results",
    sim_indices: list[int] | None = None,
    extension: str = "png",
    update_metadata: bool = True,
) -> dict[int, dict[str, float]]:
    """
    Itera sobre simulaciones y genera figuras para uno o varios tipos de plot.

    Por cada archivo ``Estado_{fase}_sim_{N}_paso_{paso}.npz`` en
    ``Results/simulation_{N}/set/`` y ``reset/``, genera una figura por cada
    tipo solicitado y la guarda en su subcarpeta propia::

        Results/simulation_{N}/Figures/state/
            Estado_{fase}_sim_{N}_paso_{paso}.{extension}
        Results/simulation_{N}/Figures/thermal_state/
            Estado_{fase}_sim_{N}_paso_{paso}.{extension}
        ...

    Subcarpeta por tipo:
        - ``"state"``       → ``state/``
        - ``"thermal"``     → ``thermal_state/``
        - ``"probability"`` → ``probability/``
        - ``"muro"``        → ``muro_termico/``

    La temperatura (``"thermal"``) se trata exactamente igual que el resto de
    tipos: se activa incluyéndola en ``plot_types``. De forma independiente,
    la temperatura máxima final de cada fase **siempre** se extrae del campo
    ``temperatura`` del npz y se escribe en la metadata (si
    ``update_metadata=True``), independientemente de si ``"thermal"`` está en
    ``plot_types`` o no.

    Args:
        plot_types:       Lista de tipos a generar. ``None`` → todos
                          (``["state", "thermal", "probability", "muro"]``).
        phases:           Fases a procesar. ``None`` → todas las 4 fases.
        results_dir:      Carpeta raíz con ``simulation_{N}/``.
        sim_indices:      Índices de simulación. ``None`` → autodetección.
        extension:        Formato de guardado (``"png"``, ``"pdf"``, ``"svg"``).
        update_metadata:  Si True escribe ``T_final_{fase}`` en ``extra`` del
                          JSON de metadatos.

    Returns:
        ``{num_sim: {"T_final_pp_set": float, ...}}``
    """
    from .Representate import RepresentateState, plot_thermal_state, RepresentateHeatmap, plot_muro_termico
    from dataclasses import replace as dc_replace

    # Validar y normalizar plot_types
    tipos_activos: tuple[PlotType, ...] = (
        tuple(TODOS_PLOT_TYPES) if plot_types is None
        else tuple(pt for pt in plot_types if pt in TIPO_SUBCARPETA)
    )
    invalidos = [pt for pt in (plot_types or []) if pt not in TIPO_SUBCARPETA]
    if invalidos:
        raise ValueError(f"plot_types no válidos: {invalidos}. Opciones: {list(TIPO_SUBCARPETA)}")

    fases_activas: tuple[str, ...] = tuple(phases) if phases is not None else TODAS_FASES
    ext = extension.lower()
    results_dir = Path(results_dir)

    # Autodetección de simulaciones
    if sim_indices is None:
        sim_indices = sorted(
            int(p.name.split("_")[-1])
            for p in results_dir.glob("simulation_*/")
            if p.is_dir() and p.name.split("_")[-1].isdigit()
        )

    resultados: dict[int, dict[str, float]] = {}

    for num_sim in sim_indices:
        sim_path = results_dir / f"simulation_{num_sim}"
        if not sim_path.is_dir():
            logger.warning(f"Carpeta {sim_path} no existe. Salto sim {num_sim}.")
            continue

        logger.info(f"=== Sim {num_sim} | tipos={list(tipos_activos)} | fases={list(fases_activas)} ===")

        # Voltajes por fase
        voltajes_por_fase = _cargar_voltaje_por_fase(sim_path, num_sim)

        # atom_size desde metadata (una sola lectura por sim)
        try:
            meta = load_metadata(sim_path, num_simulation=num_sim)
            atom_size = float(meta.params_dict.get("atom_size", 0.25e-9))
        except Exception:
            atom_size = 0.25e-9

        # Crear subcarpetas de figuras necesarias
        fig_dirs: dict[PlotType, Path] = {}
        for pt in tipos_activos:
            d = sim_path / "Figures" / TIPO_SUBCARPETA[pt]
            d.mkdir(parents=True, exist_ok=True)
            fig_dirs[pt] = d

        t_finales: dict[str, float] = {}

        for subdir_nombre in ("set", "reset"):
            subdir = sim_path / subdir_nombre
            if not subdir.is_dir():
                continue

            # Recopilar archivos Estado por fase
            archivos_por_fase: dict[str, list[tuple[int, Path]]] = {}
            for fpath in subdir.glob("Estado_*.npz"):
                resultado = _extraer_fase_y_paso(fpath.name)
                if resultado is None:
                    continue
                fase, paso = resultado
                if fase not in fases_activas:
                    continue
                archivos_por_fase.setdefault(fase, []).append((paso, fpath))

            for fase, lista in archivos_por_fase.items():
                lista_ordenada = sorted(lista, key=lambda x: x[0])
                paso_max = lista_ordenada[-1][0]

                for paso, fpath in lista_ordenada:
                    datos = np.load(fpath, allow_pickle=True)
                    claves_disponibles = set(datos.files)

                    # Campos requeridos por tipo de plot (carga lazy)
                    necesita_state  = "state" in tipos_activos or update_metadata  # actual_state siempre útil
                    necesita_temp   = "thermal" in tipos_activos or update_metadata  # T_final siempre
                    necesita_prob   = "probability" in tipos_activos
                    necesita_muro   = "muro" in tipos_activos

                    actual_state: np.ndarray = (
                        datos["actual_state"] if "actual_state" in claves_disponibles else np.zeros((1, 1), dtype=np.int64)
                    )
                    temp_data: np.ndarray = (
                        datos["temperatura"] if (necesita_temp and "temperatura" in claves_disponibles)
                        else np.array(0.0)
                    )
                    prob_matrix: np.ndarray = (
                        datos["probabilidad_matrix"] if (necesita_prob and "probabilidad_matrix" in claves_disponibles)
                        else np.zeros_like(actual_state, dtype=np.float64)
                    )
                    muro_matrix: np.ndarray = (
                        datos["matriz_para_plot_muro"] if (necesita_muro and "matriz_para_plot_muro" in claves_disponibles)
                        else np.zeros_like(actual_state, dtype=np.float64)
                    )

                    # Voltaje en este paso
                    v_arr = voltajes_por_fase.get(fase)
                    voltaje = float(v_arr[paso]) if (v_arr is not None and paso < len(v_arr)) else 0.0
                    voltaje_r = round(voltaje, 4)

                    stem = fpath.stem  # "Estado_{fase}_sim_{N}_paso_{paso}"

                    # --- Generar cada tipo de plot solicitado ---
                    for pt in tipos_activos:
                        out_path = fig_dirs[pt] / f"{stem}.{ext}"

                        if pt == "state":
                            RepresentateState(
                                matriz=actual_state,
                                voltaje=voltaje_r,
                                filename=str(out_path),
                                guardar_png=(ext == "png"),
                                atom_size=atom_size,
                            )

                        elif pt == "thermal":
                            if temp_data.ndim < 2:
                                logger.debug(f"Paso {paso} fase {fase}: temp escalar, sin mapa térmico. Salto.")
                                continue
                            types_map = _construir_types_map(actual_state, temp_data.shape)
                            plot_thermal_state(
                                T_map=temp_data,
                                types_map=types_map,
                                voltage=voltaje_r,
                                save_path=str(out_path),
                                atom_size=atom_size,
                            )

                        elif pt == "probability":
                            RepresentateHeatmap(
                                matriz=prob_matrix,
                                voltaje=voltaje_r,
                                titulo="Generation Probability",
                                filename=str(out_path),
                                cmap_name="hot",
                                label_colorbar="P (a.u.)",
                                cero_blanco=True,
                                device_size=actual_state.shape[1] * atom_size,
                            )

                        elif pt == "muro":
                            if muro_matrix.ndim < 2 or np.all(muro_matrix == 0):
                                logger.debug(f"Paso {paso} fase {fase}: sin muro térmico. Salto.")
                                continue
                            plot_muro_termico(
                                matriz_muros=muro_matrix,
                                matriz_molde=actual_state,
                                filename=str(out_path),
                                titulo=rf"Thermal Wall — $V_{{RRAM}}$ = {voltaje_r} V",
                                atom_size=atom_size,
                            )

                    # T_final siempre: temperatura del último paso de la fase
                    if paso == paso_max:
                        t_finales[f"T_final_{fase}"] = _temperatura_escalar(temp_data)

        resultados[num_sim] = t_finales

        # Actualizar metadata con T_final (siempre, independiente de plot_types)
        if update_metadata and t_finales:
            try:
                meta = load_metadata(sim_path, num_simulation=num_sim)
                extra_actualizado = {**(meta.extra or {}), **t_finales}
                save_metadata(dc_replace(meta, extra=extra_actualizado), sim_path)
                logger.info(f"Sim {num_sim}: T_final guardado → {t_finales}")
            except Exception as exc:
                logger.warning(f"Sim {num_sim}: no se pudo actualizar metadata: {exc}")

        logger.info(f"Sim {num_sim} completada. T_finales: {t_finales}")

    return resultados
