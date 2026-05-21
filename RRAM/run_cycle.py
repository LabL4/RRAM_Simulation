"""
Ejecutor del ciclo SET → RESET completo.

Toma una `SimulationConfig` (de `init_simulation.load_simulation_config`) y
ejecuta las 4 fases en orden, guardando metadatos en disco para que el
módulo `plot_results` pueda replotear sin re-ejecutar.

Robustez ante fallos
--------------------
Las fases pueden lanzar excepciones de simulación legítimas (NoPercolation,
MaxVacantes, NullResistance, FilamentosNoFormados, ...). En ese caso:

- Se persiste un `sim_metadata_{N}.json` parcial con `status` indicando dónde
  falló y los datos disponibles hasta el fallo (p.ej. voltaje_percolacion
  si PP_set llegó a percolar antes de morir).
- La excepción se relanza para que el invocador (subprocess de
  `python -m RRAM exec`) salga con código distinto de 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .init_simulation import SimulationConfig
from .persistence import (
    PHASE_ORDER,
    PHASE_PRECEDING,
    SimulationMetadata,
    load_phase_state,
    save_metadata,
    save_phase_state,
    serialize_dataclass,
)
from .phases_reset import PP_reset, SP_reset
from .phases_set import PP_set, SP_set

logger = logging.getLogger(__name__)


# Desplazamientos por defecto para anotaciones de la curva I-V. NO se persiste
# en metadata (es preferencia de plot, no propiedad de la simulación).
DESPLAZAMIENTO_IV_DEFAULT = {
    "a": (0.025, 1.0),
    "b": (+0.005, 0.27),
    "c": (0.02, 0.35),
    "d": (0.02, 1.0),
    "e": (-0.11, 0.66),
    "f": (0.025, 0.25),
    "g": (-0.12, 1),
}


@dataclass
class SimulationStates:
    """Estados finales de las 4 fases del ciclo."""
    pp_set: Optional[dict] = None
    sp_set: Optional[dict] = None
    pp_reset: Optional[dict] = None
    sp_reset: Optional[dict] = None


def _resolve_simulation_path(results_dir: Path | str, n_save: int) -> Path:
    """
    Devuelve la ruta de `Results/simulation_{N}/` consistente con
    `utils.crear_rutas_simulacion`.
    """
    return Path(results_dir) / f"simulation_{n_save}"


def _save_partial_metadata(
    cfg: SimulationConfig,
    n_save: int,
    states: SimulationStates,
    results_dir: Path | str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Persiste metadata con la mayor cantidad de datos disponibles."""
    pp_set = states.pp_set or {}
    sp_set = states.sp_set or {}
    sp_reset = states.sp_reset or {}

    voltaje_perco = pp_set.get("voltaje_percolacion", 0.0)
    creaciones = pp_set.get("creaciones_dict", {}) or {}
    roturas = sp_reset.get("roturas_dict", {}) or {}
    vecindad_inicial = sp_set.get("vecindad_inicial", None)

    meta = SimulationMetadata(
        num_simulation=n_save,
        voltaje_percolacion=float(voltaje_perco) if voltaje_perco is not None else 0.0,
        creaciones_dict=creaciones,
        roturas_dict=roturas,
        centros_calculados=list(cfg.cf_centros) if cfg.cf_centros else None,
        cf_ranges=[list(t) for t in cfg.cf_ranges],
        # Snapshot de la configuración usada para esta sim — fuente de verdad
        # para auditar/reproducir aunque cambie el CSV original.
        params_dict=serialize_dataclass(cfg.params),
        ctes_dict=serialize_dataclass(cfg.sim_ctes),
        extra={
            "status": status,
            **({"error": error} if error else {}),
            **({"vecindad_inicial": vecindad_inicial} if vecindad_inicial is not None else {}),
        },
    )
    sim_path = _resolve_simulation_path(results_dir, n_save)
    sim_path.mkdir(parents=True, exist_ok=True)
    save_metadata(meta, sim_path)


def run_cycle(
    cfg: SimulationConfig,
    results_dir: Path | str = "Results",
    start_from: Optional[str] = None,
    stop_at: Optional[str] = None,
    init_data_dir: Path | str = "Init_data",
    desplazamiento_iv: Optional[dict] = None,  # noqa: ARG001 (kept for API stability)
) -> SimulationStates:
    """
    Ejecuta SET (PP+SP) → RESET (PP+SP) y guarda metadatos.

    Args:
        cfg: Configuración cargada desde `load_simulation_config`.
        results_dir: Carpeta raíz donde cada simulación tiene su subcarpeta
            `simulation_{N+1}` (consistente con `utils.crear_rutas_simulacion`).
        start_from: Fase desde la que comenzar el ciclo. Valores válidos:
            ``"sp_set"``, ``"pp_reset"``, ``"sp_reset"``.  Si es ``None``
            (defecto) el ciclo empieza desde ``pp_set``.
        stop_at: Fase en la que terminar el ciclo (inclusive). Valores válidos:
            ``"pp_set"``, ``"sp_set"``, ``"pp_reset"``, ``"sp_reset"``. Si es
            ``None`` (defecto) el ciclo termina en ``sp_reset``.
        init_data_dir: Carpeta donde se leen/escriben los estados de fase
            (``phase_state_*.npz`` / ``.json``).  Por defecto ``"Init_data"``.
        desplazamiento_iv: Aceptado por compatibilidad pero NO persistido
            (es preferencia de plot, no estado de simulación).

    Returns:
        `SimulationStates` con los dicts de estado de las fases ejecutadas.

    Raises:
        ValueError: Si ``start_from`` o ``stop_at`` no son valores reconocidos,
            o si ``stop_at`` es anterior a ``start_from``.
        FileNotFoundError: Si se especifica ``start_from`` pero no existe el
            estado guardado de la fase precedente en ``init_data_dir``.
        Cualquier excepción de simulación legítima de las fases. Antes de
        relanzarla, persiste metadata parcial con ``status`` describiendo dónde
        falló.
    """
    _VALID_START = set(PHASE_PRECEDING.keys())
    if start_from is not None and start_from not in _VALID_START:
        raise ValueError(
            f"start_from='{start_from}' no válido. "
            f"Valores aceptados: {sorted(_VALID_START)}"
        )
    if stop_at is not None and stop_at not in PHASE_ORDER:
        raise ValueError(
            f"stop_at='{stop_at}' no válido. "
            f"Valores aceptados: {PHASE_ORDER}"
        )

    n = cfg.num_simulation
    n_save = n + 1  # offset del código original (simulation_{N+1}/)
    states = SimulationStates()

    if start_from is not None:
        preceding = PHASE_PRECEDING[start_from]
        logger.info(
            f"=== Sim {n_save} · ciclo desde '{start_from}' "
            f"(cargando estado de '{preceding}') ==="
        )
        loaded = load_phase_state(preceding, n_save, init_data_dir, cfg)
        setattr(states, preceding, loaded)
    else:
        logger.info(f"=== Sim {n_save} · ciclo SET → RESET ===")

    fases_all = [
        ("pp_set",   lambda: PP_set(
            num_simulation=n_save, params=cfg.params, sim_ctes=cfg.sim_ctes,
            CF_ranges=cfg.cf_ranges, CF_creado=cfg.cf_creado,
            CF_centros=cfg.cf_centros, actual_state=cfg.actual_state,
        )),
        ("sp_set",   lambda: SP_set(
            final_state_pp_set=states.pp_set, num_simulation=n_save,
            CF_ranges=cfg.cf_ranges,
        )),
        ("pp_reset", lambda: PP_reset(
            final_state_sp_set=states.sp_set, num_simulation=n_save,
            CF_ranges=cfg.cf_ranges,
        )),
        ("sp_reset", lambda: SP_reset(
            final_state_pp_reset=states.pp_reset, num_simulation=n_save,
            CF_ranges=cfg.cf_ranges,
        )),
    ]

    start_idx = PHASE_ORDER.index(start_from) if start_from else 0
    stop_idx  = PHASE_ORDER.index(stop_at) + 1 if stop_at else len(PHASE_ORDER)
    if stop_idx <= start_idx:
        raise ValueError(
            f"stop_at='{stop_at}' debe ser posterior a start_from='{start_from}'."
        )
    fases = fases_all[start_idx:stop_idx]

    for nombre, ejecutor in fases:
        try:
            resultado = ejecutor()
            setattr(states, nombre, resultado)
            save_phase_state(resultado, nombre, n_save, init_data_dir)
        except Exception as e:
            logger.error(f"Fase {nombre} abortada: {type(e).__name__}: {e}")
            _save_partial_metadata(
                cfg, n_save, states, results_dir,
                status=f"failed_at_{nombre}",
                error=f"{type(e).__name__}: {e}",
            )
            raise

    last_phase = fases[-1][0]
    creaciones = (states.pp_set or {}).get("creaciones_dict", {}) or {}
    roturas = (states.sp_reset or {}).get("roturas_dict", {}) or {}
    logger.info(
        f"Fase final: '{last_phase}' · "
        f"creaciones: {len(creaciones)} · roturas: {len(roturas)}"
    )

    # Todo OK → metadata completa
    _save_partial_metadata(cfg, n_save, states, results_dir, status="completed")
    return states
