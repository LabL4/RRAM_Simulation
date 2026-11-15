"""
Topología de los filamentos conductores: percolación, limpieza y clasificación por filamento.

Sustituye a la cadena el antiguo `is_path` de `Percolation` (A* de `pathfinding`) + `Clean_state_matrix`
+ `Clasificar_CF` + `Existe_filamentos` + `Eliminar_filamentos_incompletos` (grafos de `networkx`)
por un etiquetado de componentes conexas con `scipy.ndimage.label` y una poda de hojas sobre
arrays. Reproduce exactamente sus resultados (ver tests/test_topologia_filamentos.py), en una sola
pasada y unas 40 veces más rápido.

Reglas (las mismas que el código al que sustituye):

1. Una celda es conductora si su valor es >= 1. Vecindad: los 4 vecinos ortogonales.
2. **Percola** si alguna componente conexa toca la columna 0 y la última (los dos electrodos).
3. **Matriz limpia**: solo esas componentes, podando repetidamente las celdas con un único vecino
   que no están en las columnas de electrodo (ramas muertas). La poda nunca desconecta nada.
4. **Existe el filamento i** si alguna celda de la matriz limpia de la columna 0 cae en su banda de
   filas `cf_ranges[i]` (el filamento se asigna por su contacto con el electrodo izquierdo).
5. **Matriz de filamentos**: la matriz limpia con a cero todas las filas de las bandas cuyo
   filamento no existe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class TopologiaFilamentos:
    """Resultado del análisis de un estado. Las matrices son de solo lectura (se comparten con la caché)."""

    #: Alguna componente une los dos electrodos.
    percola: bool
    #: Componentes que unen los electrodos, sin ramas muertas (antes `Clean_state_matrix`).
    matriz_limpia: np.ndarray
    #: Un booleano por filamento (antes `Existe_filamentos`).
    existe: List[bool]
    #: Matriz limpia sin las bandas de los filamentos que no existen (antes `Eliminar_filamentos_incompletos`).
    matriz_filamentos: np.ndarray


def filas_banda_filamentos(n_filas: int, centros, grosor) -> np.ndarray:
    """
    Máscara 1D (n_filas,) de las filas de la banda de cada filamento: centro ± grosor.

    `grosor` es un entero común o una lista con un valor por filamento. Es la definición única de
    la banda: la usan la máscara de generación (`Generation.create_custom_mask`), las filas de la
    interfaz donde nacen los oxígenos (`Recombination.filas_interfaz_filamento`) y el perfil del
    electrodo (`thermal.muro.filas_filamento`).
    """
    grosores = [int(grosor)] * len(centros) if isinstance(grosor, (int, np.integer)) else [int(g) for g in grosor]
    if len(grosores) != len(centros):
        raise ValueError("Debe haber un grosor asignado para cada centro de filamento.")
    filas = np.zeros(n_filas, dtype=bool)
    for centro, g in zip(centros, grosores):
        filas[max(0, int(centro) - g) : min(n_filas, int(centro) + g + 1)] = True
    return filas


def _componentes_entre_electrodos(conductora: np.ndarray) -> np.ndarray:
    """Máscara de las celdas cuya componente conexa (4-vecindad) toca las columnas 0 y última."""
    etiquetas, _ = ndimage.label(conductora)
    izquierda = np.unique(etiquetas[:, 0])
    derecha = np.unique(etiquetas[:, -1])
    validas = np.intersect1d(izquierda[izquierda > 0], derecha[derecha > 0], assume_unique=True)
    if validas.size == 0:
        return np.zeros(conductora.shape, dtype=bool)
    return np.isin(etiquetas, validas)


def _podar_ramas_muertas(mascara: np.ndarray) -> np.ndarray:
    """
    Quita a la vez, y hasta que no quede ninguna, las celdas con exactamente un vecino que no están
    en las columnas de electrodo. Igual que la poda de hojas por grado del grafo original.
    """
    m = mascara.copy()
    while True:
        grado = np.zeros(m.shape, dtype=np.int8)
        grado[1:, :] += m[:-1, :]
        grado[:-1, :] += m[1:, :]
        grado[:, 1:] += m[:, :-1]
        grado[:, :-1] += m[:, 1:]
        hojas = m & (grado == 1)
        hojas[:, 0] = False
        hojas[:, -1] = False
        if not hojas.any():
            return m
        m &= ~hojas


def percola(estado: np.ndarray) -> bool:
    """True si alguna componente conductora une la columna 0 con la última."""
    return bool(_componentes_entre_electrodos(np.asarray(estado) >= 1).any())


#: Nombre antiguo (`is_path` de `Percolation`), aún usado desde notebooks.
is_path = percola


def analizar_topologia(estado: np.ndarray, cf_ranges: Sequence[Tuple[int, int]]) -> TopologiaFilamentos:
    """Percolación, matriz limpia, existencia por filamento y matriz de filamentos de `estado`."""
    estado = np.asarray(estado)
    validas = _componentes_entre_electrodos(estado >= 1)

    if not validas.any():
        vacia = np.zeros(estado.shape, dtype=int)
        vacia.setflags(write=False)
        return TopologiaFilamentos(False, vacia, [False] * len(cf_ranges), vacia)

    limpia = _podar_ramas_muertas(validas)

    filas_electrodo = np.flatnonzero(limpia[:, 0])
    cubiertas = np.zeros(estado.shape[0], dtype=bool)
    for fila_min, fila_max in cf_ranges:
        cubiertas[fila_min : fila_max + 1] = True
    sin_banda = filas_electrodo[~cubiertas[filas_electrodo]]
    if sin_banda.size:
        raise ValueError(f"Nodo {(int(sin_banda[0]), 0)} no pudo asignarse a ningún filamento")

    existe = [bool(limpia[fila_min : fila_max + 1, 0].any()) for fila_min, fila_max in cf_ranges]

    filamentos = limpia.copy()
    for (fila_min, fila_max), ok in zip(cf_ranges, existe):
        if not ok:
            filamentos[fila_min : fila_max + 1, :] = False

    matriz_limpia = limpia.astype(int)
    matriz_filamentos = filamentos.astype(int)
    matriz_limpia.setflags(write=False)
    matriz_filamentos.setflags(write=False)
    return TopologiaFilamentos(True, matriz_limpia, existe, matriz_filamentos)


class CacheTopologia:
    """
    Recuerda el último análisis y lo reutiliza mientras el estado no cambie.

    En la mayoría de pasos no se genera ni se recombina ninguna vacante, así que el análisis del
    paso anterior sigue valiendo. Comparar dos matrices cuesta mucho menos que analizarlas.
    """

    def __init__(self, cf_ranges: Sequence[Tuple[int, int]]):
        self.cf_ranges = [tuple(r) for r in cf_ranges]
        self._estado: np.ndarray | None = None
        self._resultado: TopologiaFilamentos | None = None
        self.calculos = 0
        self.reutilizados = 0

    def analizar(self, estado: np.ndarray) -> TopologiaFilamentos:
        if self._resultado is not None and np.array_equal(estado, self._estado):
            self.reutilizados += 1
            return self._resultado
        self._resultado = analizar_topologia(estado, self.cf_ranges)
        self._estado = np.array(estado, copy=True)
        self.calculos += 1
        return self._resultado
