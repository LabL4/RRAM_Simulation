"""
Facade del módulo Simulation.

Reexporta las clases y funciones públicas del núcleo de simulación tras la
factorización en módulos independientes. Mantiene compatibilidad hacia atrás
con código que importa `from RRAM import Simulation` y accede a símbolos como
`Simulation.PP_set`, `Simulation.SimulationParameters`, etc.

Estructura tras factorización:
    - parameters.py            → SimulationParameters
    - constants_simulation.py  → SimulationConstants
    - decorators.py            → medir_tiempo
    - filament_tracking.py     → procesar_filamentos_creados,
                                 procesar_filamentos_destruidos,
                                 actualizar_parametros_por_filamento
    - state_updates.py         → update_state_generation,
                                 update_state_recombinate
    - phases_set.py            → PP_set, SP_set
    - phases_reset.py          → PP_reset, SP_reset
    - iv_analysis.py           → simulation_IV
"""

from .constants_simulation import SimulationConstants
from .decorators import medir_tiempo
from .filament_tracking import (
    actualizar_parametros_por_filamento,
    procesar_filamentos_creados,
    procesar_filamentos_destruidos,
)
from .iv_analysis import simulation_IV
from .parameters import SimulationParameters
from .phases_reset import PP_reset, SP_reset
from .phases_set import PP_set, SP_set
from .state_updates import update_state_generation, update_state_recombinate

__all__ = [
    "SimulationParameters",
    "SimulationConstants",
    "medir_tiempo",
    "procesar_filamentos_creados",
    "procesar_filamentos_destruidos",
    "actualizar_parametros_por_filamento",
    "update_state_generation",
    "update_state_recombinate",
    "PP_set",
    "SP_set",
    "PP_reset",
    "SP_reset",
    "simulation_IV",
]
