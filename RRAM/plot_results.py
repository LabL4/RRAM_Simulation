"""
Postprocesado y plot a partir de datos en disco.

Lee `Data_*.npz` y `sim_metadata_{N}.json` que dejó `run_cycle` en
`Results/simulation_{N}/`. NO requiere haber ejecutado el ciclo en la misma
sesión: replotear es independiente de la ejecución.

Uso programático:

    from RRAM.plot_results import plot_results
    plot_results(num_simulation=3)

O vía CLI: `python -m RRAM plot 3`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .iv_analysis import simulation_IV
from .persistence import load_metadata
from .run_cycle import DESPLAZAMIENTO_IV_DEFAULT

logger = logging.getLogger(__name__)


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
