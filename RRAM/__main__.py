"""
Entry point del paquete RRAM.

Subcomandos:
    init         Pre-genera Init_data/init_state_*.npz para todas las simulaciones.
    exec         Ejecuta el ciclo SET → RESET de una simulación concreta.
    plot         Replotea la curva I-V sin marcar (I-V_{N}.png) leyendo del disco.
    plot_marcado Replotea la curva I-V con puntos a-g marcados (I-V_marcado_{N}.png).
    all          init (si falta) → exec → plot + plot_marcado. Comportamiento histórico.

Uso:
    python -m RRAM init
    python -m RRAM exec          <num_simulation>
    python -m RRAM plot          <num_simulation>
    python -m RRAM plot_marcado  <num_simulation>
    python -m RRAM all           <num_simulation> [--guardar-datos]

Variables de entorno:
    RRAM_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR  (default INFO)
"""

from __future__ import annotations
import numpy as np
import argparse
import logging
import os
import sys

# Backend sin pantalla. Es una variable de entorno que matplotlib lee cuando se importe, de modo que
# `exec` no carga matplotlib; solo se carga al dibujar (plot, plot_marcado, all).
os.environ.setdefault("MPLBACKEND", "Agg")

from .init_simulation import build_initial_states, load_simulation_config
from .logging_config import setup_logging
from .run_cycle import run_cycle


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m RRAM",
        description="Simulación RRAM: init → exec → plot.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Pre-genera estados iniciales en Init_data/")
    p_init.add_argument("--init-data-dir", default="Init_data")
    p_init.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Siembra la generación del estado inicial de cada simulación i con "
            "seed+i, para que Init_data sea reproducible entre corridas "
            "(distinto por sim, idéntico corrida a corrida). "
            "Por defecto: aleatorio real (sin fijar)."
        ),
    )
    p_init.add_argument(
        "--mismo-estado-inicial",
        action="store_true",
        default=False,
        help=(
            "Con --seed fijado, usa la MISMA semilla (sin +i) para todas las "
            "simulaciones en vez de seed+i. Si además comparten geometría "
            "(n_filas, n_cols, num_trampas, filamentos), el estado inicial "
            "(posición de trampas) es idéntico en todo el ensemble. Útil para "
            "aislar el efecto de un parámetro del barrido controlando el resto. "
            "Por defecto: False (seed+i, ensemble con estados distintos)."
        ),
    )

    # exec
    p_exec = sub.add_parser("exec", help="Ejecuta el ciclo SET → RESET.")
    p_exec.add_argument("num_simulation", type=int)
    p_exec.add_argument(
        "--num-filamentos", type=int, default=None, help="Sobreescribe num_filamentos del CSV para esta ejecución."
    )
    p_exec.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Fija la semilla de numpy.random al inicio de cada fase (PP_set, SP_set, "
            "PP_reset, SP_reset) para que la corrida sea exactamente reproducible. "
            "Por defecto: aleatorio real (sin fijar)."
        ),
    )
    p_exec.add_argument("--init-data-dir", default="Init_data")
    p_exec.add_argument("--results-dir", default="Results")
    p_exec.add_argument(
        "--start-from",
        choices=["sp_set", "pp_reset", "sp_reset"],
        default=None,
        metavar="FASE",
        help=(
            "Fase desde la que comenzar el ciclo (sp_set | pp_reset | sp_reset). "
            "Requiere que exista el estado guardado de la fase precedente en Init_data/."
        ),
    )
    p_exec.add_argument(
        "--stop-at",
        choices=["pp_set", "sp_set", "pp_reset", "sp_reset"],
        default=None,
        metavar="FASE",
        help="Fase en la que terminar el ciclo, inclusive (pp_set | sp_set | pp_reset | sp_reset).",
    )
    p_exec.add_argument(
        "--no-muro",
        action="store_true",
        default=False,
        help="Desactiva el muro térmico en todas las fases del ciclo, SET y RESET (pasa matriz_muros=None al solver).",
    )
    p_exec.add_argument(
        "--sin-estados",
        action="store_true",
        default=False,
        help="No guarda estados intermedios (Estado_*): solo datos, estados finales y metadata.",
    )

    # plot
    p_plot = sub.add_parser("plot", help="Replotea la curva I-V sin marcar (I-V_{N}.png).")
    p_plot.add_argument("num_simulation", type=int, help="Índice 0-based de la simulación en el CSV (el mismo que en exec).")
    p_plot.add_argument("--results-dir", default="Results")
    p_plot.add_argument(
        "--no-forma-onda",
        action="store_true",
        default=False,
        help="No genera las figuras V-t/I-t de las etapas con forma de onda (V-t_I-t_{fase}_{N}.png).",
    )

    p_plot.add_argument(
        "--sin-experimental",
        action="store_true",
        default=False,
        help="No superpone el ciclo experimental en la curva I-V.",
    )

    # plot_marcado
    p_plot_marcado = sub.add_parser(
        "plot_marcado", help="Replotea la curva I-V con puntos a-g marcados (I-V_marcado_{N}.png)."
    )
    p_plot_marcado.add_argument("num_simulation", type=int, help="Índice 0-based de la simulación en el CSV (el mismo que en exec).")
    p_plot_marcado.add_argument("--results-dir", default="Results")
    p_plot_marcado.add_argument(
        "--puntos-iv",
        default=None,
        metavar="JSON",
        help="Archivo JSON con la tabla de puntos marcados (formato de iv_analysis.PUNTOS_IV). Por defecto: PUNTOS_IV.",
    )
    p_plot_marcado.add_argument(
        "--sin-experimental",
        action="store_true",
        default=False,
        help="No superpone el ciclo experimental en la curva I-V.",
    )

    # all (compat con el flujo histórico)
    p_all = sub.add_parser("all", help="init (si falta) + exec + plot.")
    p_all.add_argument("num_simulation", type=int)
    p_all.add_argument(
        "--num-filamentos", type=int, default=None, help="Sobreescribe num_filamentos del CSV para esta ejecución."
    )
    p_all.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Fija la semilla de numpy.random al inicio de cada fase (PP_set, SP_set, "
            "PP_reset, SP_reset) para que la ejecucion sea exactamente reproducible. "
            "Por defecto: aleatorio real (sin fijar)."
        ),
    )
    p_all.add_argument(
        "--mismo-estado-inicial",
        action="store_true",
        default=False,
        help=(
            "Con --seed fijado, usa la MISMA semilla (sin +i) al generar el "
            "estado inicial de todas las simulaciones (ver 'init --mismo-estado-inicial'). "
            "Solo tiene efecto si esta sim dispara la generación de Init_data."
        ),
    )
    p_all.add_argument(
        "--sin-estados",
        action="store_true",
        default=False,
        help="No guarda estados intermedios (Estado_*): solo datos, estados finales y metadata.",
    )
    p_all.add_argument("--guardar-datos", action="store_true")
    p_all.add_argument("--init-data-dir", default="Init_data")
    p_all.add_argument("--results-dir", default="Results")
    p_all.add_argument(
        "--puntos-iv",
        default=None,
        metavar="JSON",
        help="Archivo JSON con la tabla de puntos marcados (formato de iv_analysis.PUNTOS_IV). Por defecto: PUNTOS_IV.",
    )
    p_all.add_argument(
        "--start-from",
        choices=["sp_set", "pp_reset", "sp_reset"],
        default=None,
        metavar="FASE",
        help=(
            "Fase desde la que comenzar el ciclo (sp_set | pp_reset | sp_reset). "
            "Requiere que exista el estado guardado de la fase precedente en Init_data/."
        ),
    )
    p_all.add_argument(
        "--stop-at",
        choices=["pp_set", "sp_set", "pp_reset", "sp_reset"],
        default=None,
        metavar="FASE",
        help="Fase en la que terminar la simulacion, inclusive (pp_set | sp_set | pp_reset | sp_reset).",
    )
    p_all.add_argument(
        "--no-forma-onda",
        action="store_true",
        default=False,
        help="No genera las figuras V-t/I-t de las etapas con forma de onda (V-t_I-t_{fase}_{N}.png).",
    )

    p_all.add_argument(
        "--sin-experimental",
        action="store_true",
        default=False,
        help="No superpone el ciclo experimental en la curva I-V.",
    )

    return p


def _cmd_init(args) -> int:

    setup_logging(num_simulation=None, to_console=True)
    build_initial_states(
        init_data_dir=args.init_data_dir,
        seed=args.seed,
        mismo_estado_inicial=args.mismo_estado_inicial,
    )
    return 0


def _cmd_exec(args) -> int:
    setup_logging(num_simulation=args.num_simulation + 1)
    log = logging.getLogger("RRAM.__main__")
    try:
        cfg = load_simulation_config(
            num_simulation=args.num_simulation,
            init_data_dir=args.init_data_dir,
            num_filamentos=args.num_filamentos,
            seed=args.seed,
        )
        run_cycle(
            cfg,
            results_dir=args.results_dir,
            start_from=args.start_from,
            stop_at=args.stop_at,
            init_data_dir=args.init_data_dir,
            usar_muro=not args.no_muro,
            guardar_estados=not args.sin_estados,
        )
        return 0
    except Exception as e:
        log.exception(f"exec sim={args.num_simulation + 1} abortado: {type(e).__name__}: {e}")
        return 1


def _cmd_plot(args) -> int:
    from .plot_results import plot_results  # import perezoso: arrastra matplotlib

    n_save = args.num_simulation + 1  # carpeta Results/simulation_{N+1}, igual que exec
    setup_logging(num_simulation=n_save, file_mode="a")
    log = logging.getLogger("RRAM.__main__")
    try:
        plot_results(
            num_simulation=n_save,
            results_dir=args.results_dir,
            plot_forma_onda=not args.no_forma_onda,
            mostrar_experimental=not args.sin_experimental,
        )
        return 0
    except FileNotFoundError as e:
        log.error(f"plot sim={n_save}: {e}")
        return 2
    except Exception as e:
        log.exception(f"plot sim={n_save} abortado: {e}")
        return 1


def _leer_puntos_iv(ruta: str | None) -> dict | None:
    """Tabla de puntos marcados desde un JSON, o None para usar PUNTOS_IV."""
    if ruta is None:
        return None
    import json

    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def _cmd_plot_marcado(args) -> int:
    from .plot_results import plot_results_marcado  # import perezoso: arrastra matplotlib

    n_save = args.num_simulation + 1  # carpeta Results/simulation_{N+1}, igual que exec
    setup_logging(num_simulation=n_save, file_mode="a")
    log = logging.getLogger("RRAM.__main__")
    try:
        plot_results_marcado(
            num_simulation=n_save,
            results_dir=args.results_dir,
            puntos_iv=_leer_puntos_iv(args.puntos_iv),
            mostrar_experimental=not args.sin_experimental,
        )
        return 0
    except FileNotFoundError as e:
        log.error(f"plot_marcado sim={n_save}: {e}")
        return 2
    except Exception as e:
        log.exception(f"plot_marcado sim={n_save} abortado: {e}")
        return 1


def _cmd_all(args) -> int:
    from .plot_results import plot_results, plot_results_marcado  # import perezoso: arrastra matplotlib

    setup_logging(num_simulation=args.num_simulation + 1)
    logger = logging.getLogger("RRAM.__main__")

    # 1. Init: solo si no existe el estado correspondiente
    from pathlib import Path

    init_state_path = Path(args.init_data_dir) / f"init_state_{args.num_simulation}.npz"
    if not init_state_path.is_file():
        logger.info(f"init_state ausente ({init_state_path}); generando todos los iniciales.")
        build_initial_states(
            init_data_dir=args.init_data_dir,
            seed=args.seed,
            mismo_estado_inicial=args.mismo_estado_inicial,
        )
    else:
        logger.info(f"init_state ya existe ({init_state_path}); saltando init.")

    # 2. Exec
    try:
        cfg = load_simulation_config(
            num_simulation=args.num_simulation,
            init_data_dir=args.init_data_dir,
            num_filamentos=args.num_filamentos,
            seed=args.seed,
        )
        run_cycle(
            cfg,
            results_dir=args.results_dir,
            start_from=args.start_from,
            stop_at=args.stop_at,
            init_data_dir=args.init_data_dir,
            guardar_estados=not args.sin_estados,
        )
    except Exception as e:
        logger.exception(f"all/exec sim={args.num_simulation + 1} abortado: {e}")
        return 1

    # 3. Plot (solo si exec terminó bien) — genera las dos figuras, como antes
    #    de dividir `plot`/`plot_marcado` en subcomandos separados.
    try:
        plot_results(
            num_simulation=args.num_simulation + 1,
            results_dir=args.results_dir,
            plot_forma_onda=not args.no_forma_onda,
            mostrar_experimental=not args.sin_experimental,
        )
        plot_results_marcado(
            num_simulation=args.num_simulation + 1,
            results_dir=args.results_dir,
            puntos_iv=_leer_puntos_iv(args.puntos_iv),
            mostrar_experimental=not args.sin_experimental,
        )
    except FileNotFoundError as e:
        logger.error(f"all/plot sim={args.num_simulation + 1}: {e}")
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dispatch = {
        "init": _cmd_init,
        "exec": _cmd_exec,
        "plot": _cmd_plot,
        "plot_marcado": _cmd_plot_marcado,
        "all": _cmd_all,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
