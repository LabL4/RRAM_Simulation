"""
Procesos estocásticos que cambian el estado de vacantes al final de cada paso.

- `ProcesoGeneracion` (SET): genera vacantes hasta un cupo.
- `ProcesoRecombinacion` (RESET): genera iones de oxígeno, los mueve y los recombina con vacantes.

Cada proceso guarda su propia configuración y su propio estado (cupo alcanzado; iones, relojes y
campo de probabilidad), de modo que el bucle de la etapa solo le pide "actualiza el estado".
La física vive en `state_updates`, `Generation` y `Recombination`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .acumulador_tiempo import RelojesIones
from .state_updates import evaluar_campo_probabilidad, update_state_generation, update_state_recombinate

logger = logging.getLogger(__name__)


class ProcesoGeneracion:
    """Generación de vacantes con cupo máximo. Cuando el cupo se alcanza deja de sortear."""

    def __init__(
        self,
        factor_vecinos: float,
        factor_libre: float,
        max_vacantes: int,
        neighbor_mode: str,
        mascara: Optional[np.ndarray] = None,
        aviso_cupo: str = "Se ha alcanzado la ocupación máxima",
        rng=None,
    ):
        self.rng = rng
        self.factor_vecinos = factor_vecinos
        self.factor_libre = factor_libre
        self.max_vacantes = max_vacantes
        self.neighbor_mode = neighbor_mode
        #: Máscara que limita dónde se generan vacantes (None = toda la malla).
        self.mascara = mascara
        self.aviso_cupo = aviso_cupo
        #: True desde el primer paso en que no se sortea por estar el cupo lleno.
        self.cupo_alcanzado = False

    def actualizar(self, actual_state, total_vacantes, params, sim_ctes, E_filas, temperatura, k: int, tiempo_efectivo: float):
        """Devuelve (estado, probabilidades). Con el cupo lleno no sortea: (estado sin tocar, None)."""
        if total_vacantes < self.max_vacantes:
            return update_state_generation(
                actual_state,
                params,
                sim_ctes,
                E_filas,
                temperatura,
                self.factor_vecinos,
                self.factor_libre,
                self.max_vacantes,
                neighbor_mode=self.neighbor_mode,
                custom_mask=self.mascara,
                num_iteracion=k,
                tiempo_efectivo=tiempo_efectivo,
                rng=self.rng,
            )
        if not self.cupo_alcanzado:
            logger.info(f"{self.aviso_cupo} en el paso {k}.")
            self.cupo_alcanzado = True
        return actual_state, None


class ProcesoRecombinacion:
    """Iones de oxígeno del RESET: generación por voltaje, movimiento con reloj propio y recombinación."""

    def __init__(
        self,
        oxygen_state: np.ndarray,
        relojes: RelojesIones,
        oxygen_config: dict,
        velocity_thresholds: dict,
        filas_permitidas: np.ndarray,
        max_oxigenos: int,
        rng=None,
    ):
        self.rng = rng
        self.oxygen_state = oxygen_state
        self.relojes = relojes
        #: {voltaje umbral: máximo de oxígenos por paso}, ordenado de mayor a menor umbral.
        self.oxygen_config = oxygen_config
        #: {voltaje umbral: velocidad de los iones [m/s]}.
        self.velocity_thresholds = velocity_thresholds
        #: Filas de la interfaz donde se pueden generar oxígenos.
        self.filas_permitidas = filas_permitidas
        #: Máximo de iones simultáneos.
        self.max_oxigenos = max_oxigenos
        self._campo_probabilidad = None

    def empezar_paso(self, voltage: float) -> None:
        """Avanza el reloj de los iones que ya existen (antes de generar: los nuevos empiezan en 0)."""
        self.relojes.actualizar(voltage, self.oxygen_state)

    def actualizar(self, actual_state, voltage, E_filas, temperatura, params, sim_ctes, tiempo_efectivo: float):
        """Genera, mueve y recombina. Devuelve el estado de vacantes; los iones quedan en `oxygen_state`."""
        actual_state, self.oxygen_state, self._campo_probabilidad = update_state_recombinate(
            voltage=voltage,
            E_field_filas=E_filas,
            oxygen_config=self.oxygen_config,
            sim_ctes=sim_ctes,
            params=params,
            actual_state=actual_state,
            oxygen_state=self.oxygen_state,
            temperatura=temperatura,
            velocity_thresholds=self.velocity_thresholds,
            relojes=self.relojes,
            filas_permitidas=self.filas_permitidas,
            max_oxigenos=self.max_oxigenos,
            tiempo_efectivo=tiempo_efectivo,
            rng=self.rng,
        )
        return actual_state

    def probabilidad(self):
        """Campo de probabilidad de recombinación del último paso (None si aún no hubo ninguno)."""
        return evaluar_campo_probabilidad(self._campo_probabilidad)
