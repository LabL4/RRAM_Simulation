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

from dataclasses import dataclass, field, replace as dc_replace
from typing import List, Optional, Tuple
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
    """Configuración completa lista para ejecutar el ciclo SET→RESET.

    cf_ranges, cf_centros y cf_creado se calculan automáticamente en
    __post_init__ a partir de params y sim_ctes — fuente única de verdad.
    """

    num_simulation: int
    params: SimulationParameters
    sim_ctes: SimulationConstants
    actual_state: np.ndarray
    num_trampas: int

    cf_ranges: List[tuple] = field(init=False)
    cf_centros: List[int] = field(init=False)
    cf_creado: np.ndarray = field(init=False)

    def __post_init__(self):
        cf_ranges, _, cf_centros = utils.generar_configuracion_filamentos(
            eje_x=self.params.y_size,
            eje_y=self.params.x_size,
            num_filamentos=self.sim_ctes.num_filamentos,
            grosores_filamento=self.sim_ctes.grosor_filamento,
            centros_override=self.sim_ctes.centros_filamento,  # ← nuevo
        )
        self.cf_ranges = cf_ranges
        self.cf_centros = cf_centros
        self.cf_creado = np.full(len(cf_ranges), False, dtype=bool)


# ---------------------------------------------------------------------------
# 1. Pre-generación de estados iniciales (todas las simulaciones del CSV)
# ---------------------------------------------------------------------------


def build_initial_states(
    init_data_dir: Path | str = "Init_data",
    seed: Optional[int] = None,
) -> int:
    """
    Pre-genera `init_state_{i}.npz` para todas las simulaciones del CSV.

    Args:
        init_data_dir: Carpeta con los CSVs, donde se escriben los `.npz`.
        seed: Si se especifica, cada simulación `i` siembra np.random con
            `seed + i` justo antes de sortear sus trampas, de forma que el
            estado inicial sea reproducible entre corridas sin que todas las
            simulaciones arranquen del mismo estado (el ensemble se conserva).
            None (defecto) = aleatorio real, comportamiento histórico.
    """
    init_data_dir = Path(init_data_dir)
    archivo_params = init_data_dir / "simulation_parameters.csv"
    if not archivo_params.is_file():
        raise FileNotFoundError(
            f"No se encuentra {archivo_params}. Lanza ConfigManager.export_to_init_data() en el notebook primero."
        )

    df_params = pd.read_csv(archivo_params)
    num_simulations = len(df_params)

    archivo_ctes = init_data_dir / "simulation_constants.csv"
    df_ctes = pd.read_csv(archivo_ctes) if archivo_ctes.is_file() else None

    logger.info(f"Construyendo estados iniciales para {num_simulations} simulaciones...")

    for i, row in df_params.iterrows():
        # Semilla por simulación: `seed + i` mantiene cada init_state distinto
        # entre sims pero idéntico corrida a corrida. Sin seed no se toca
        # np.random (aleatorio real).
        if seed is not None:
            np.random.seed(seed + int(i))

        eje_x = int(math.ceil(row["device_size_y"] / row["atom_size"]))
        eje_y = int(math.ceil(row["device_size_x"] / row["atom_size"]))
        num_trampas = int(row["num_trampas"])

        num_filamentos = 2
        grosor_filamento = None
        centros_filamento = None
        if df_ctes is not None and i < len(df_ctes):
            raw_nf = df_ctes.iloc[i].get("num_filamentos", None)
            if raw_nf is not None:
                try:
                    num_filamentos = int(float(raw_nf))
                except (ValueError, TypeError):
                    pass

            raw_gf = df_ctes.iloc[i].get("grosor_filamento", None)
            if raw_gf is not None:
                try:
                    grosor_filamento = ast.literal_eval(str(raw_gf).strip())
                except (ValueError, SyntaxError):
                    grosor_filamento = int(float(raw_gf))

            raw_cf = df_ctes.iloc[i].get("centros_filamento", None)
            if raw_cf is not None:
                try:
                    parsed = ast.literal_eval(str(raw_cf).strip())
                    if isinstance(parsed, list):
                        centros_filamento = parsed
                except (ValueError, SyntaxError):
                    pass

        f_ranges, regiones_pesos, _ = utils.generar_configuracion_filamentos(
            eje_x,
            eje_y,
            num_filamentos=num_filamentos,
            grosores_filamento=grosor_filamento,
            centros_override=centros_filamento,
        )
        init_state = Generation.initial_state_priv(eje_x, eje_y, num_trampas, regiones_pesos)

        logger.info(
            f"Simulación {i}: dispositivo=({eje_x},{eje_y}) "
            f"trampas={num_trampas} filamentos={num_filamentos} ranges={f_ranges} grosor={grosor_filamento}"
        )

        out = init_data_dir / f"init_state_{i}.npz"
        np.savez_compressed(out, actual_state=init_state)

    origen = f"seed={seed} (por sim: seed+i)" if seed is not None else "aleatorio real"
    logger.info(f"{num_simulations} estados iniciales generados en {init_data_dir} · {origen}.")
    return num_simulations


# ---------------------------------------------------------------------------
# 2. Carga de la config de una simulación concreta
# ---------------------------------------------------------------------------


def load_simulation_config(
    num_simulation: int,
    init_data_dir: Path | str = "Init_data",
    num_filamentos: Optional[int] = None,
    seed: Optional[int] = None,
) -> SimulationConfig:
    """
    Lee CSVs + estado inicial para una simulación concreta.

    num_filamentos se lee de simulation_constants.csv.  Si se proporciona
    explícitamente, sobreescribe el valor del CSV (útil para experimentos
    puntuales sin regenerar el CSV).  SimulationConfig deriva
    cf_ranges/cf_centros/cf_creado automáticamente en __post_init__.

    Args:
        num_simulation:  Índice de la simulación dentro del CSV (0-based).
        init_data_dir:   Carpeta con los CSVs e init_state_{i}.npz.
        num_filamentos:  Override opcional sobre el valor del CSV.
        seed:            Override opcional de semilla (--seed en la CLI). Si se
            especifica, cada fase (PP_set, SP_set, PP_reset, SP_reset) reinicia
            np.random con este valor al empezar, para reproducibilidad exacta
            entre corridas. None (defecto) = aleatorio real.

    Returns:
        SimulationConfig listo para `run_cycle`.
    """
    init_data_dir = Path(init_data_dir)

    sim_parmtrs = utils.read_csv_to_dic(str(init_data_dir / "simulation_parameters.csv"))
    params = SimulationParameters.from_dict(sim_parmtrs[num_simulation])

    if seed is not None and seed != params.seed:
        logger.info(f"seed override: {params.seed} → {seed}")
        params = dc_replace(params, seed=seed)

    sim_cte = utils.read_csv_to_dic(str(init_data_dir / "simulation_constants.csv"))
    ctes = SimulationConstants.from_dict(sim_cte[num_simulation])

    if num_filamentos is not None and num_filamentos != ctes.num_filamentos:
        logger.info(f"num_filamentos override: CSV={ctes.num_filamentos} → {num_filamentos}")
        grosor = ctes.grosor_filamento
        if isinstance(grosor, list) and len(grosor) > num_filamentos:
            grosor = grosor[:num_filamentos]
            logger.info(f"grosor_filamento truncado a {num_filamentos} elemento(s): {grosor}")
        ctes = dc_replace(ctes, num_filamentos=num_filamentos, grosor_filamento=grosor)

    init_state_path = init_data_dir / f"init_state_{num_simulation}"
    actual_state = utils.cargar_estado(init_state_path)

    num_trampas = int(sim_parmtrs[num_simulation].get("num_trampas", 0) or 0)

    cfg = SimulationConfig(
        num_simulation=num_simulation,
        params=params,
        sim_ctes=ctes,
        actual_state=actual_state,
        num_trampas=num_trampas,
    )

    logger.info(
        f"Config cargada · sim={num_simulation} · trampas={num_trampas} · "
        f"filamentos={ctes.num_filamentos} · ranges={cfg.cf_ranges} · centros={cfg.cf_centros} · "
        f"Energia activacion={ctes.generation_energy} · Energia recombinacion={ctes.recombination_energy}"
    )

    return cfg
