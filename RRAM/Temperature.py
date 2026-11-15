"""Fachada de compatibilidad: el modelo térmico vive en `RRAM.thermal` (joule, amortiguacion, fvm, muro).
"""

from .thermal.joule import Temperature_Joule  # noqa: F401
from .thermal.amortiguacion import actualizar_temperatura_electrodo, amortiguacion_activa, amortiguar_enfriamiento, amortiguar_mapa_fvm, como_mapa, extra_sin_amortiguar, temperatura_electrodo_joule  # noqa: F401
from .thermal.muro import calcular_filas_intermedias, calcular_perfiles_muro, colocar_muro_termico, componer_matriz_muro_plot, extraer_perfiles_filamentos, extraer_perfiles_temperatura, filas_filamento  # noqa: F401
from .thermal.fvm import ID_ELECTRODO, ID_OXIDO, SolverTermico, _SOLVER_POR_DEFECTO, _ensamblar_matriz_termica, _factorizar, _k_efectiva, calculate_heat_source, crear_matriz_materiales, obtener_centro_CF, solve_thermal_state  # noqa: F401
