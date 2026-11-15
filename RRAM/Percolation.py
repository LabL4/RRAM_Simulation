"""
Compatibilidad: `is_path` se sigue usando desde notebooks (crear_estado_*, Preparar_pp_reset*).

La implementación vive en `topologia_filamentos`, con etiquetado de componentes conexas en lugar
del A* de la librería `pathfinding`. Mismo resultado (tests/test_topologia_filamentos.py).
"""

import numpy as np

from .topologia_filamentos import percola


def is_path(configuration_matrix: np.ndarray) -> bool:
    """True si las celdas con valor >= 1 conectan la primera columna con la última (4-vecindad)."""
    return percola(configuration_matrix)
