"""
Esquema de columnas de `datos_sim` (los `Data_<fase>_<N>.npz`).

Es la única definición del orden de columnas: la usa el grabador para escribir (y para la cabecera
`header_datos_sim` que va dentro de cada fichero) y cualquier lector puede importarla para saber
dónde está cada magnitud sin contar posiciones a mano.

Si cambia el orden o el significado de una columna, subir `VERSION_ESQUEMA`.
"""

from __future__ import annotations

from typing import Dict, List

VERSION_ESQUEMA = 1


def columnas_datos(N: int, con_t_ef: bool = False) -> List[str]:
    """
    Columnas de `datos_sim`: [t, V, I_total, R_total] + I_1..I_N + R_1..R_N + T_1..T_N (+ t_ef en el RESET).
    El orden por bloques es el de `PasoSimulacion.fila`.
    """
    cols = ["t[s]", "V[V]", "I_total[A]", "R_total[Ohm]"]
    cols += [f"I_{i}[A]" for i in range(1, N + 1)]
    cols += [f"R_{i}[Ohm]" for i in range(1, N + 1)]
    cols += [f"T_{i}[K]" for i in range(1, N + 1)]
    if con_t_ef:
        cols += ["t_ef[s]"]
    return cols


def indices_columnas(N: int, con_t_ef: bool = False) -> Dict[str, int]:
    """Posición de cada columna por nombre, p. ej. `indices_columnas(2)["R_total[Ohm]"] == 3`."""
    return {nombre: i for i, nombre in enumerate(columnas_datos(N, con_t_ef))}
