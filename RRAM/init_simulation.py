"""
Inicialización de simulaciones RRAM.

Dos responsabilidades:
1. **`build_initial_states`**: pre-genera las matrices iniciales
   `Init_data/init_state_{i}.npz` para todas las simulaciones del CSV.
   (Reemplaza el script raíz `Init_simulation.py`).
2. **`load_simulation_config`**: para una simulación concreta, lee CSV de
   parámetros + constantes + estado inicial y devuelve un `SimulationConfig`
   listo para `run_cycle`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import math
import ast

import pandas as pd
import numpy as np

from .constants_simulation import SimulationConstants
from .parameters import SimulationParameters
from . import Generation, utils

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuración completa lista para ejecutar el ciclo SET→RESET."""

    num_simulation: int
    params: SimulationParameters
    sim_ctes: SimulationConstants
    cf_ranges: List[tuple]
    cf_centros: List[int]
    cf_creado: np.ndarray
    actual_state: np.ndarray
    num_trampas: int


# ---------------------------------------------------------------------------
# 1. Pre-generación de estados iniciales (todas las simulaciones del CSV)
# ---------------------------------------------------------------------------


def build_initial_states(
    init_data_dir: Path | str = "Init_data",
    num_filamentos_para_pesos: int = 2,
) -> int:
    init_data_dir = Path(init_data_dir)
    archivo_params = init_data_dir / "simulation_parameters.csv"
    if not archivo_params.is_file():
        raise FileNotFoundError(
            f"No se encuentra {archivo_params}. Lanza ConfigManager.export_to_init_data() en el notebook primero."
        )

    df_params = pd.read_csv(archivo_params)
    num_simulations = len(df_params)

    # Leer constantes para obtener grosor_filamento por simulación
    archivo_ctes = init_data_dir / "simulation_constants.csv"
    df_ctes = pd.read_csv(archivo_ctes) if archivo_ctes.is_file() else None

    logger.info(f"Construyendo estados iniciales para {num_simulations} simulaciones...")

    for i, row in df_params.iterrows():
        eje_x = int(math.ceil(row["device_size_y"] / row["atom_size"]))
        eje_y = int(math.ceil(row["device_size_x"] / row["atom_size"]))
        num_trampas = int(row["num_trampas"])

        # Extraer grosor_filamento de las constantes si está disponible
        grosor_filamento = None
        if df_ctes is not None and i < len(df_ctes):
            raw = df_ctes.iloc[i].get("grosor_filamento", None)
            if raw is not None:
                try:
                    grosor_filamento = ast.literal_eval(str(raw).strip())
                except (ValueError, SyntaxError):
                    grosor_filamento = int(float(raw))

        f_ranges, regiones_pesos, _ = utils.generar_configuracion_filamentos(
            eje_x,
            eje_y,
            num_filamentos=num_filamentos_para_pesos,
            grosores_filamento=grosor_filamento,
        )
        init_state = Generation.initial_state_priv(eje_x, eje_y, num_trampas, regiones_pesos)

        logger.info(
            f"Simulación {i}: dispositivo=({eje_x},{eje_y}) "
            f"trampas={num_trampas} ranges={f_ranges} grosor={grosor_filamento}"
        )

        out = init_data_dir / f"init_state_{i}.npz"
        np.savez_compressed(out, actual_state=init_state)

    logger.info(f"{num_simulations} estados iniciales generados en {init_data_dir}.")
    return num_simulations


# ---------------------------------------------------------------------------
# 2. Carga de la config de una simulación concreta
# ---------------------------------------------------------------------------


def load_simulation_config(
    num_simulation: int,
    init_data_dir: Path | str = "Init_data",
    num_filamentos: Optional[int] = None,
) -> SimulationConfig:
    """
    Lee CSVs + estado inicial para una simulación concreta.

    Args:
        num_simulation: Índice de la simulación dentro del CSV (0-based).
        init_data_dir: Carpeta con `simulation_parameters.csv`,
            `simulation_constants.csv` y `init_state_{i}.npz`.
        num_filamentos: Si se proporciona, sobreescribe `ctes.num_filamentos`.

    Returns:
        SimulationConfig listo para `run_cycle`.
    """
    init_data_dir = Path(init_data_dir)

    sim_parmtrs = utils.read_csv_to_dic(str(init_data_dir / "simulation_parameters.csv"))
    params = SimulationParameters.from_dict(sim_parmtrs[num_simulation])

    sim_cte = utils.read_csv_to_dic(str(init_data_dir / "simulation_constants.csv"))
    ctes = SimulationConstants.from_dict(sim_cte[num_simulation])

    n_fil = num_filamentos if num_filamentos is not None else ctes.num_filamentos

    cf_ranges, _, cf_centros = utils.generar_configuracion_filamentos(
        eje_x=params.y_size,
        eje_y=params.x_size,
        num_filamentos=n_fil,
    )
    cf_creado = np.full(len(cf_ranges), False, dtype=bool)

    init_state_path = init_data_dir / f"init_state_{num_simulation}"
    actual_state = utils.cargar_estado(init_state_path)

    num_trampas = int(sim_parmtrs[num_simulation].get("num_trampas", 0) or 0)

    logger.info(
        f"Config cargada · sim={num_simulation} · trampas={num_trampas} · "
        f"filamentos={n_fil} · ranges={cf_ranges} · centros={cf_centros}"
    )

    return SimulationConfig(
        num_simulation=num_simulation,
        params=params,
        sim_ctes=ctes,
        cf_ranges=cf_ranges,
        cf_centros=cf_centros,
        cf_creado=cf_creado,
        actual_state=actual_state,
        num_trampas=num_trampas,
    )
