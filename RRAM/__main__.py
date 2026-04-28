"""
Entry point del paquete RRAM.

Subcomandos:
    init        Pre-genera Init_data/init_state_*.npz para todas las simulaciones.
    exec        Ejecuta el ciclo SET → RESET de una simulación concreta.
    plot        Replotea una simulación leyendo del disco (no re-ejecuta).
    all         init (si falta) → exec → plot. Comportamiento histórico.

Uso:
    python -m RRAM init
    python -m RRAM exec  <num_simulation> [--num-filamentos N]
    python -m RRAM plot  <num_simulation>
    python -m RRAM all   <num_simulation> [--num-filamentos N] [--guardar-datos]

Variables de entorno:
    RRAM_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR  (default INFO)
"""

from __future__ import annotations

import argparse
import logging
import sys

import matplotlib

matplotlib.use("Agg")

from .init_simulation import build_initial_states, load_simulation_config
from .logging_config import setup_logging
from .plot_results import plot_results
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
    p_init.add_argument("--num-filamentos-pesos", type=int, default=2)

    # exec
    p_exec = sub.add_parser("exec", help="Ejecuta el ciclo SET → RESET.")
    p_exec.add_argument("num_simulation", type=int)
    p_exec.add_argument("--num-filamentos", type=int, default=None)
    p_exec.add_argument("--init-data-dir", default="Init_data")
    p_exec.add_argument("--results-dir", default="Results")

    # plot
    p_plot = sub.add_parser("plot", help="Replotea una simulación previamente ejecutada.")
    p_plot.add_argument("num_simulation", type=int, help="Índice usado al ejecutar (offset +1).")
    p_plot.add_argument("--results-dir", default="Results")

    # all (compat con el flujo histórico)
    p_all = sub.add_parser("all", help="init (si falta) + exec + plot.")
    p_all.add_argument("num_simulation", type=int)
    p_all.add_argument("--num-filamentos", type=int, default=None)
    p_all.add_argument("--guardar-datos", action="store_true")
    p_all.add_argument("--init-data-dir", default="Init_data")
    p_all.add_argument("--results-dir", default="Results")

    return p


def _cmd_init(args) -> int:
    setup_logging(num_simulation=None, to_console=True)
    build_initial_states(
        init_data_dir=args.init_data_dir,
        num_filamentos_para_pesos=args.num_filamentos_pesos,
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
        )
        run_cycle(cfg, results_dir=args.results_dir)
        return 0
    except Exception as e:
        # Las fases pueden lanzar excepciones de simulación legítimas
        # (NoPercolation, MaxVacantes...). Las logueamos con traceback en el
        # archivo de log y propagamos código de error al subprocess.
        log.exception(
            f"exec sim={args.num_simulation + 1} abortado: {type(e).__name__}: {e}"
        )
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


def _cmd_all(args) -> int:
    setup_logging(num_simulation=args.num_simulation + 1)
    logger = logging.getLogger("RRAM.__main__")

    # 1. Init: solo si no existe el estado correspondiente
    from pathlib import Path

    init_state_path = Path(args.init_data_dir) / f"init_state_{args.num_simulation}.npz"
    if not init_state_path.is_file():
        logger.info(f"init_state ausente ({init_state_path}); generando todos los iniciales.")
        build_initial_states(init_data_dir=args.init_data_dir)
    else:
        logger.info(f"init_state ya existe ({init_state_path}); saltando init.")

    # 2. Exec
    try:
        cfg = load_simulation_config(
            num_simulation=args.num_simulation,
            init_data_dir=args.init_data_dir,
            num_filamentos=args.num_filamentos,
        )
        run_cycle(cfg, results_dir=args.results_dir)
    except Exception as e:
        logger.exception(f"all/exec sim={args.num_simulation + 1} abortado: {e}")
        return 1

    # 3. Plot (solo si exec terminó bien)
    try:
        plot_results(
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
        "all": _cmd_all,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
