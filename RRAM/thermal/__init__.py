"""Modelo térmico: Joule, amortiguación, solver FVM y muro térmico.

Reexporta la API pública de sus cuatro módulos (`RRAM.Temperature` es un alias de este paquete).
Los nombres privados (`_SOLVER_POR_DEFECTO`, `_factorizar`, ...) se importan de `RRAM.thermal.fvm`.
"""

from .amortiguacion import (
    actualizar_temperatura_electrodo,
    amortiguacion_activa,
    amortiguar_enfriamiento,
    amortiguar_mapa_fvm,
    como_mapa,
    temperatura_electrodo_joule,
)
from .fvm import ID_ELECTRODO, ID_OXIDO, SolverTermico, crear_matriz_materiales, obtener_centro_CF, solve_thermal_state
from .joule import Temperature_Joule
from .muro import (
    calcular_filas_intermedias,
    calcular_perfiles_muro,
    colocar_muro_termico,
    componer_matriz_muro_plot,
    extraer_perfiles_filamentos,
    extraer_perfiles_temperatura,
    filas_filamento,
)

__all__ = [
    "Temperature_Joule",
    "actualizar_temperatura_electrodo", "amortiguacion_activa", "amortiguar_enfriamiento", "amortiguar_mapa_fvm",
    "como_mapa", "temperatura_electrodo_joule",
    "calcular_filas_intermedias", "calcular_perfiles_muro", "colocar_muro_termico", "componer_matriz_muro_plot",
    "extraer_perfiles_filamentos", "extraer_perfiles_temperatura", "filas_filamento",
    "ID_ELECTRODO", "ID_OXIDO", "SolverTermico", "crear_matriz_materiales", "obtener_centro_CF", "solve_thermal_state",
]
