"""
Lo que calcula un paso de simulación y que no debe sobrevivir al siguiente.

Las fases creaban estas magnitudes como variables locales sueltas y las guardaban en disco con
`locals().get(...)`, de modo que un paso que no calculaba una de ellas (p. ej. sin percolación)
guardaba la del paso anterior. Un `PasoSimulacion` nuevo por iteración lo hace imposible: lo que el
paso no calcula se queda en None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import Temperature


@dataclass
class PasoSimulacion:
    #: Mapa de resistencias por celda (solo si percola).
    R_local: Optional[np.ndarray] = None
    #: Matriz de filamentos completos usada en el paso (fases de RESET; en SET vive en la fase).
    cf_clean_matrix: Optional[np.ndarray] = None
    #: Muro térmico en el marco interior (solo si se coloca).
    matriz_muro: Optional[np.ndarray] = None
    #: Perfil de temperatura anterior de cada filamento (solo si se extrae).
    perfiles: Optional[list] = None
    #: Temperatura antes de amortiguar (solo en las ramas que amortiguan).
    temperatura_sin_amortiguar: object = None
    #: Matriz de probabilidades del sorteo de generación (SET).
    probabilidad_generacion: Optional[np.ndarray] = None
    #: Resistencia total y valores por filamento de la fila de datos.
    R_total: float = np.nan
    I_fils: Optional[List[float]] = None
    R_fils: Optional[List[float]] = None
    T_fils: Optional[List[float]] = None

    def registrar_ohm(self, ohm) -> None:
        """Resultado de la rama eléctrica percolante (`fisica_paso.ResultadoOhmico`)."""
        self.R_local = ohm.R_local
        self.R_total = ohm.R_total
        self.I_fils = ohm.I_fils
        self.R_fils = ohm.R_fils

    def sin_percolacion(self, N: int) -> None:
        """No percola: sin resistencia ni corriente por filamento."""
        self.R_total = np.nan
        self.I_fils = [np.nan] * N
        self.R_fils = [np.nan] * N

    def fila(self, t: float, V: float, I_total: float, *extra) -> list:
        """Fila de datos: [t, V, I_total, R_total] + I_fils + R_fils + T_fils + extra."""
        return [t, V, I_total, self.R_total] + self.I_fils + self.R_fils + self.T_fils + list(extra)

    def matriz_muro_para_plot(self, centros) -> Optional[np.ndarray]:
        return Temperature.componer_matriz_muro_plot(self.matriz_muro, centros, self.perfiles)
