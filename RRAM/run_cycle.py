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

import numpy as np

from .estado_fase import EstadoFase
from .init_simulation import SimulationConfig
from .persistence import (
    PHASE_ORDER,
    PHASE_PRECEDING,
    SimulationMetadata,
    cargar_estado_fase,
    save_metadata,
    save_phase_state,
    serialize_dataclass,
)
from .phases_reset import PP_reset, SP_reset
from .phases_set import PP_set, SP_set

logger = logging.getLogger(__name__)


@dataclass
class SimulationStates:
    """Estados finales (`EstadoFase`) de las 4 fases del ciclo."""
    pp_set: Optional[EstadoFase] = None
    sp_set: Optional[EstadoFase] = None
    pp_reset: Optional[EstadoFase] = None
    sp_reset: Optional[EstadoFase] = None


def _resolve_simulation_path(results_dir: Path | str, n_save: int) -> Path:
    """
    Devuelve la ruta de `Results/simulation_{N}/` consistente con
    `grabador.crear_rutas_simulacion`.
    """
    return Path(results_dir) / f"simulation_{n_save}"


def _temperaturas_max_por_fase(states: SimulationStates) -> dict:
    """
    Recopila la temperatura máxima de cada filamento en cada fase ejecutada.

    Cada fase acumula su propio `T_max_fils` (pico sobre todos sus pasos, contando
    solo los que resuelven el FVM). Aquí se juntan las 4 en la forma que acaba en
    la metadata::

        {"pp_set": {"T_1": 735.2, "T_2": 690.1}, "sp_set": {...}}

    Las fases no ejecutadas no aparecen. Un `None` (fase que nunca resolvió el
    FVM para ese filamento) se guarda como `null`.
    """
    temperaturas_max: dict = {}
    for fase in PHASE_ORDER:
        estado = getattr(states, fase)
        T_max_fils = None if estado is None else estado.T_max_fils
        if T_max_fils is None:
            continue
        temperaturas_max[fase] = {
            f"T_{i}": (float(t) if t is not None else None) for i, t in enumerate(T_max_fils, start=1)
        }
    return temperaturas_max


def _guardar_estado_fallo(error: Exception) -> None:
    """
    Guarda en disco el estado que dejó una excepción de simulación (si lo trae) y su figura.

    Las excepciones (`exceptions.ExcepcionConEstado`) solo transportan los datos; aquí, en la
    capa que las captura, se escribe. Un fallo al guardar el diagnóstico se registra pero no
    tapa el error original.
    """
    ruta = getattr(error, "ruta_estado_fallo", None)
    estado = getattr(error, "actual_state", None)
    if ruta is None or estado is None:
        return
    try:
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ruta, actual_state=estado)
        logger.info(f"Estado del fallo guardado en {ruta}")
        figura = getattr(error, "ruta_figura_fallo", None)
        if figura is not None:
            from . import Representate  # import perezoso: solo hace falta al dibujar

            Path(figura).parent.mkdir(parents=True, exist_ok=True)
            Representate.RepresentateState(estado, round(error.voltage, 3), str(figura))
    except Exception as e2:  # noqa: BLE001 - el diagnóstico no debe ocultar el fallo original
        logger.warning(f"No se pudo guardar el estado del fallo: {type(e2).__name__}: {e2}")


#: Sufijo del estado de una fase que se paró por excepción (Init_data/phase_state_N_<fase>_fallida.*).
#: `--start-from` no lo usa nunca: solo sirve para inspeccionar lo que la fase llegó a calcular.
SUFIJO_FASE_FALLIDA = "_fallida"


def _guardar_estado_fase_fallida(estado_fase, nombre: str, n_save: int, init_data_dir) -> None:
    """Guarda las variables de una fase que se paró por excepción. Un fallo aquí no tapa el original."""
    try:
        save_phase_state(estado_fase, f"{nombre}{SUFIJO_FASE_FALLIDA}", n_save, init_data_dir)
    except Exception as e2:  # noqa: BLE001
        logger.warning(f"No se pudieron guardar las variables de la fase fallida {nombre}: {type(e2).__name__}: {e2}")


def _save_partial_metadata(
    cfg: SimulationConfig,
    n_save: int,
    states: SimulationStates,
    results_dir: Path | str,
    status: str,
    error: Optional[str] = None,
    usar_muro: bool = True,
    start_from: Optional[str] = None,
    stop_at: Optional[str] = None,
) -> None:
    """Persiste metadata con la mayor cantidad de datos disponibles.

    Los flags que cambian la física de la corrida (`usar_muro`, `seed`) y el
    tramo del ciclo realmente ejecutado (`start_from`, `stop_at`) se guardan en
    `extra`: sin ellos no hay forma de saber a posteriori con qué modelo se
    generó una carpeta de resultados.
    """
    pp_set, sp_set = states.pp_set, states.sp_set

    voltaje_perco = pp_set.voltaje_percolacion if pp_set else 0.0
    paso_perco = pp_set.paso_percolacion if pp_set else None
    creaciones = (pp_set.creaciones_dict if pp_set else None) or {}
    # SP_reset recibe y amplía el roturas_dict de PP_reset: se toma el más avanzado
    # disponible, para no perder las roturas si el ciclo se detiene en pp_reset.
    ultimo_reset = states.sp_reset or states.pp_reset
    roturas = (ultimo_reset.roturas_dict if ultimo_reset else None) or {}
    # La ponen los notebooks que preparan un estado de sp_set (Preparar_pp_reset*)
    vecindad_inicial = sp_set.extra.get("vecindad_inicial") if sp_set else None

    params_dict = serialize_dataclass(cfg.params)

    meta = SimulationMetadata(
        num_simulation=n_save,
        voltaje_percolacion=float(voltaje_perco) if voltaje_perco is not None else 0.0,
        paso_percolacion=int(paso_perco) if paso_perco is not None else None,
        creaciones_dict=creaciones,
        roturas_dict=roturas,
        centros_calculados=list(cfg.cf_centros) if cfg.cf_centros else None,
        cf_ranges=[list(t) for t in cfg.cf_ranges],
        # Snapshot de la configuración usada para esta sim — fuente de verdad
        # para auditar/reproducir aunque cambie el CSV original.
        params_dict=params_dict,
        ctes_dict=serialize_dataclass(cfg.sim_ctes),
        # Todo el voltaje: lo pedido (configuracion) y lo que pasó en cada etapa
        # ejecutada (ejecucion), también si la etapa falló a mitad.
        protocolo_voltaje=cfg.protocolo.informe(),
        extra={
            "status": status,
            # Tramo del ciclo realmente ejecutado. Se guardan las fases efectivas,
            # no los argumentos crudos: `--start-from` sin valor significa pp_set y
            # `--stop-at` sin valor significa sp_reset.
            "start_from": start_from or PHASE_ORDER[0],
            "stop_at": stop_at or PHASE_ORDER[-1],
            # Flags que cambian la física de la corrida.
            "muro_termico": bool(usar_muro),
            "seed": cfg.params.seed,
            # Pico de temperatura de cada filamento, por fase.
            "temperaturas_max": _temperaturas_max_por_fase(states),
            # Pasos con estado intermedio guardado en cada fase ejecutada
            "estados_guardados": {f: list(getattr(states, f).estados_guardados)
                                  for f in PHASE_ORDER[PHASE_ORDER.index(start_from or PHASE_ORDER[0]):]
                                  if getattr(states, f) is not None},
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
    usar_muro: bool = True,
    guardar_estados: bool = True,
) -> SimulationStates:
    """
    Ejecuta SET (PP+SP) → RESET (PP+SP) y guarda metadatos.

    Args:
        cfg: Configuración cargada desde `load_simulation_config`.
        results_dir: Carpeta raíz donde cada simulación tiene su subcarpeta
            `simulation_{N+1}` (consistente con `grabador.crear_rutas_simulacion`).
        start_from: Fase desde la que comenzar el ciclo. Valores válidos:
            ``"sp_set"``, ``"pp_reset"``, ``"sp_reset"``.  Si es ``None``
            (defecto) el ciclo empieza desde ``pp_set``.
        stop_at: Fase en la que terminar el ciclo (inclusive). Valores válidos:
            ``"pp_set"``, ``"sp_set"``, ``"pp_reset"``, ``"sp_reset"``. Si es
            ``None`` (defecto) el ciclo termina en ``sp_reset``.
        init_data_dir: Carpeta donde se leen/escriben los estados de fase
            (``phase_state_*.npz`` / ``.json``).  Por defecto ``"Init_data"``.

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
        loaded = cargar_estado_fase(preceding, n_save, init_data_dir, cfg)
        setattr(states, preceding, loaded)
    else:
        logger.info(f"=== Sim {n_save} · ciclo SET → RESET ===")

    # Cada cuántos pasos se guarda un estado intermedio; 0 = ninguno (--sin-estados)
    cada_estados = cfg.sim_ctes.pasos_guardar_estado if guardar_estados else 0

    fases_all = [
        ("pp_set",   lambda: PP_set(
            num_simulation=n_save, params=cfg.params, sim_ctes=cfg.sim_ctes,
            CF_ranges=cfg.cf_ranges, CF_creado=cfg.cf_creado, protocolo=cfg.protocolo,
            num_pasos_guardar_estado=cada_estados,
            CF_centros=cfg.cf_centros, actual_state=cfg.actual_state,
            usar_muro=usar_muro, results_dir=results_dir, init_data_dir=init_data_dir,
        )),
        ("sp_set",   lambda: SP_set(
            final_state_pp_set=states.pp_set, num_simulation=n_save,
            CF_ranges=cfg.cf_ranges, protocolo=cfg.protocolo,
            num_pasos_guardar_estado=cada_estados,
            usar_muro=usar_muro, results_dir=results_dir,
        )),
        ("pp_reset", lambda: PP_reset(
            final_state_sp_set=states.sp_set, num_simulation=n_save,
            CF_ranges=cfg.cf_ranges, protocolo=cfg.protocolo,
            num_pasos_guardar_estado=cada_estados,
            usar_muro=usar_muro, results_dir=results_dir,
        )),
        ("sp_reset", lambda: SP_reset(
            final_state_pp_reset=states.pp_reset, num_simulation=n_save,
            CF_ranges=cfg.cf_ranges, protocolo=cfg.protocolo,
            num_pasos_guardar_estado=cada_estados,
            results_dir=results_dir,
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
            _guardar_estado_fallo(e)
            estado_fase = getattr(e, "estado_fase", None)
            if estado_fase is not None:
                # Lo que llegó a calcular la fase: entra en la metadata parcial (voltaje de
                # percolación, creaciones, roturas, T máximas) y se guarda aparte.
                setattr(states, nombre, estado_fase)
                _guardar_estado_fase_fallida(estado_fase, nombre, n_save, init_data_dir)
            _save_partial_metadata(
                cfg, n_save, states, results_dir,
                status=f"failed_at_{nombre}",
                error=f"{type(e).__name__}: {e}",
                usar_muro=usar_muro, start_from=start_from, stop_at=stop_at,
            )
            raise

    last_phase = fases[-1][0]
    creaciones = (states.pp_set.creaciones_dict if states.pp_set else None) or {}
    roturas = (states.sp_reset.roturas_dict if states.sp_reset else None) or {}
    logger.info(
        f"Fase final: '{last_phase}' · "
        f"creaciones: {len(creaciones)} · roturas: {len(roturas)}"
    )

    # Todo OK → metadata completa
    _save_partial_metadata(
        cfg, n_save, states, results_dir, status="completed",
        usar_muro=usar_muro, start_from=start_from, stop_at=stop_at,
    )
    return states
