"""
Física de un paso común a las cuatro fases cuando el sistema percola.

Estos bloques estaban copiados casi literalmente en PP_set, SP_set, PP_reset y SP_reset:
- rama eléctrica: mapa de resistencias (geometría + temperatura anterior) y corriente de Ohm,
- muro térmico: perfiles de los muros entre filamentos y su colocación,
- rama térmica: mapa de materiales, fuentes de calor y solver FVM.
Las decisiones que sí cambian entre fases (si hay muro, si se amortigua) se quedan en cada fase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import CurrentSolver, Temperature, exceptions


@dataclass(frozen=True, eq=False)
class ResultadoOhmico:
    R_local: np.ndarray
    I_total: float
    R_total: float
    I_fils: List[float]
    R_fils: List[float]


def resolver_rama_electrica(
    voltage: float,
    cf_matrix: np.ndarray,
    temperatura_anterior,
    params,
    sim_ctes,
    CF_ranges,
    *,
    simulation_path,
    num_simulation: int,
    actual_state: np.ndarray,
) -> ResultadoOhmico:
    """
    Mapa de resistencias por celda y corriente de Ohm (filamentos en paralelo).

    El mapa combina la geometría del filamento con la temperatura del paso anterior y es la entrada
    única de las ramas eléctrica y térmica, de modo que no pueden desincronizarse.
    Una resistencia nula lanza `NullResistanceException`.
    """
    R_local = CurrentSolver.mapa_resistencias(
        cf_matrix=cf_matrix,
        temperatura=temperatura_anterior,
        sigma_0=sim_ctes.sigma_0,
        alpha_T=sim_ctes.alpha_T,
        T_0=params.init_temp,
        delta_z_r=sim_ctes.delta_z_r,
    )
    try:
        I_total, R_total, I_fils, R_fils = CurrentSolver.OmhCurrent_filamentos(voltage, R_local, CF_ranges)
    except ZeroDivisionError:
        raise exceptions.NullResistanceException(
            simulation_path=simulation_path,
            voltage=voltage,
            num_simulation=num_simulation,
            actual_state=actual_state,
        )
    return ResultadoOhmico(R_local, I_total, R_total, I_fils, R_fils)


def construir_muro_termico(cf_matrix, perfiles, filas_intermedias, dist_casillas, params, sim_ctes) -> np.ndarray:
    """
    Muro térmico (marco interior) entre filamentos: perfil de cada muro a partir del de su filamento y
    colocación en las filas intermedias, solo sobre óxido. El molde es `cf_matrix`, la misma matriz
    que genera el mapa de materiales y las fuentes de calor, para que el muro sea coherente con el solver.
    """
    perfiles_muros = Temperature.calcular_perfiles_muro(
        perfiles_filamentos=perfiles,
        distancias_casillas=dist_casillas,
        pendiente_temperatura=sim_ctes.pendiente_temperatura,
        atom_size=params.atom_size,
        T_ambient=params.init_temp,
    )
    return Temperature.colocar_muro_termico(
        matriz_molde=cf_matrix,
        filas_intermedias=filas_intermedias,
        perfiles_muros_calculados=perfiles_muros,
    )


def resolver_temperatura_fvm(
    cf_matrix: np.ndarray,
    R_local: np.ndarray,
    I_fils: List[float],
    T_electrodo,
    params,
    sim_ctes,
    CF_ranges,
    matriz_muro: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Mapa térmico estacionario (Ny, Nx+2), SIN amortiguar.

    Materiales a partir de la geometría, calor de cada filamento según su propia corriente y, si se
    da `matriz_muro` (marco interior), muro térmico como condición Dirichlet (columnas de electrodo a 0).
    """
    materials_map = Temperature.crear_matriz_materiales(cf_matrix)
    Q_source_map = Temperature.calculate_heat_source(
        atom_size=params.atom_size,
        delta_z_r=sim_ctes.delta_z_r,
        R_local=R_local,
        factor_generar_calor=sim_ctes.factor_generar_calor,
        CF_ranges=CF_ranges,
        I_fils=I_fils,
    )
    matriz_muros_fvm = None
    if matriz_muro is not None:
        columna_ceros = np.zeros((matriz_muro.shape[0], 1))
        matriz_muros_fvm = np.hstack([columna_ceros, matriz_muro, columna_ceros])
    return Temperature.solve_thermal_state(
        types_map=materials_map,
        Q_map=Q_source_map,
        thermal_props=sim_ctes.propiedades_termicas,
        atom_size=params.atom_size,
        T_electrodo=T_electrodo,
        matriz_muros=matriz_muros_fvm,
    )
