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
import sys

import matplotlib

matplotlib.use("Agg")

from .init_simulation import build_initial_states, load_simulation_config
from .logging_config import setup_logging
from .plot_results import plot_results, plot_results_marcado
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

    # plot
    p_plot = sub.add_parser("plot", help="Replotea la curva I-V sin marcar (I-V_{N}.png).")
    p_plot.add_argument("num_simulation", type=int, help="Índice usado al ejecutar (offset +1).")
    p_plot.add_argument("--results-dir", default="Results")

    # plot_marcado
    p_plot_marcado = sub.add_parser(
        "plot_marcado", help="Replotea la curva I-V con puntos a-g marcados (I-V_marcado_{N}.png)."
    )
    p_plot_marcado.add_argument("num_simulation", type=int, help="Índice usado al ejecutar (offset +1).")
    p_plot_marcado.add_argument("--results-dir", default="Results")

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
    p_all.add_argument("--guardar-datos", action="store_true")
    p_all.add_argument("--init-data-dir", default="Init_data")
    p_all.add_argument("--results-dir", default="Results")
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

    return p


def _cmd_init(args) -> int:

    setup_logging(num_simulation=None, to_console=True)
    build_initial_states(init_data_dir=args.init_data_dir, seed=args.seed)
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
        )
        return 0
    except Exception as e:
        log.exception(f"exec sim={args.num_simulation + 1} abortado: {type(e).__name__}: {e}")
        return 1


def _cmd_plot(args) -> int:
    setup_logging(num_simulation=args.num_simulation, file_mode="a")
    log = logging.getLogger("RRAM.__main__")
    try:
        plot_results(num_simulation=args.num_simulation, results_dir=args.results_dir)
        return 0
    except FileNotFoundError as e:
        log.error(f"plot sim={args.num_simulation}: {e}")
        return 2
    except Exception as e:
        log.exception(f"plot sim={args.num_simulation} abortado: {e}")
        return 1


def _cmd_plot_marcado(args) -> int:
    setup_logging(num_simulation=args.num_simulation, file_mode="a")
    log = logging.getLogger("RRAM.__main__")
    try:
        plot_results_marcado(num_simulation=args.num_simulation, results_dir=args.results_dir)
        return 0
    except FileNotFoundError as e:
        log.error(f"plot_marcado sim={args.num_simulation}: {e}")
        return 2
    except Exception as e:
        log.exception(f"plot_marcado sim={args.num_simulation} abortado: {e}")
        return 1


def _cmd_all(args) -> int:
    setup_logging(num_simulation=args.num_simulation + 1)
    logger = logging.getLogger("RRAM.__main__")

    # 1. Init: solo si no existe el estado correspondiente
    from pathlib import Path

    init_state_path = Path(args.init_data_dir) / f"init_state_{args.num_simulation}.npz"
    if not init_state_path.is_file():
        logger.info(f"init_state ausente ({init_state_path}); generando todos los iniciales.")
        build_initial_states(init_data_dir=args.init_data_dir, seed=args.seed)
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
        )
        plot_results_marcado(
            num_simulation=args.num_simulation + 1,
            results_dir=args.results_dir,
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
